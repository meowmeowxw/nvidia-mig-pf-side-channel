#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import threading
import signal
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =====================================================================================
# 1. Backbone (Feature Extractor)
# =====================================================================================

class FeatureExtractor(nn.Module):
    """Base CNN model that acts as a feature extractor (backbone)."""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.output_channels = 64
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

# =====================================================================================
# 2. Classification Heads (The Transfer Learning Layers)
# =====================================================================================

class ResidualHead(nn.Module):
    """Standard convolutional residual block head."""
    def __init__(self, in_channels, num_classes):
        super(ResidualHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        out = self.pool(out).view(out.size(0), -1)
        return self.classifier(out)

class SEHead(nn.Module):
    """Squeeze-and-Excitation head, in its original 2D form."""
    def __init__(self, in_channels, num_classes, reduction=16):
        super(SEHead, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out = self.pool(out).view(out.size(0), -1)
        return self.classifier(out)

class TransformerHead(nn.Module):
    """A head based on a Transformer Encoder layer, as used in Vision Transformers."""
    def __init__(self, in_channels, num_classes, nhead=4, num_layers=2):
        super(TransformerHead, self).__init__()
        self.in_channels = in_channels
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(in_channels, num_classes)
        self.pos_embedding = nn.Parameter(torch.randn(1, 8*8, in_channels))

    def forward(self, x):
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class NonLocalHead(nn.Module):
    """Non-Local Neural Network block head in its original 2D form."""
    def __init__(self, in_channels, num_classes, inter_channels=None):
        super(NonLocalHead, self).__init__()
        if inter_channels is None: inter_channels = in_channels // 2
        self.inter_channels = max(inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0); nn.init.constant_(self.W.bias, 0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = F.softmax(torch.matmul(theta_x, phi_x), dim=-1)
        y = torch.matmul(f, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        z = self.W(y) + x
        out = self.pool(z).view(z.size(0), -1)
        return self.classifier(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): return self.sigmoid(self.conv1(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))

class CBAMHead(nn.Module):
    def __init__(self, in_channels, num_classes, reduction=16):
        super(CBAMHead, self).__init__()
        self.ca = ChannelAttention(in_channels, ratio=reduction)
        self.sa = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return self.classifier(self.pool(x).view(x.size(0), -1))

class SpatialTransformerHead(nn.Module):
    """Spatial Transformer Network head in its original 2D form."""
    def __init__(self, in_channels, num_classes):
        super(SpatialTransformerHead, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1) # Flatten
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        out = self.pool(x_transformed).view(x.size(0), -1)
        return self.classifier(out)


# =====================================================================================
# 3. Main Model and Runner (Unchanged)
# =====================================================================================

class TransferLearningModel(nn.Module):
    def __init__(self, backbone, head, freeze_backbone=True):
        super(TransferLearningModel, self).__init__()
        self.backbone = backbone
        self.head = head
        if freeze_backbone:
            for param in self.backbone.parameters(): param.requires_grad = False
    def forward(self, x): return self.head(self.backbone(x))
    def get_trainable_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def get_frozen_params(self): return sum(p.numel() for p in self.parameters() if not p.requires_grad)

class LargeRandomDataset(Dataset):
    def __init__(self, total_samples, image_size=(3, 32, 32), num_classes=10):
        self.total_samples, self.image_size, self.num_classes = total_samples, image_size, num_classes
    def __len__(self): return self.total_samples
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        return torch.randn(self.image_size), torch.randint(0, self.num_classes, (1,)).item()

class StreamingInferenceRunner:
    def __init__(self, model, device, dataset, batch_size=32):
        self.model, self.device, self.dataset, self.batch_size = model, device, dataset, batch_size
        self.running = False; self.inference_count = 0; self.batch_count = 0
        self.total_transfer_time = 0; self.total_inference_time = 0
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    def start_inference(self):
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop); self.thread.daemon = True; self.thread.start()
        print(f"Started streaming inference on {self.device}")
    def stop_inference(self):
        self.running = False
        if hasattr(self, 'thread'): self.thread.join()
        avg_transfer = self.total_transfer_time / max(1, self.batch_count) * 1000
        avg_inference = self.total_inference_time / max(1, self.batch_count) * 1000
        print(f"\nStopped inference. Total batches: {self.batch_count:,}, Total samples: {self.inference_count:,}")
        print(f"  Avg Transfer: {avg_transfer:.2f} ms/batch | Avg Inference: {avg_inference:.2f} ms/batch")
    def _inference_loop(self):
        data_iter = iter(self.dataloader)
        while self.running:
            try:
                transfer_start = time.time()
                try: batch_data, _ = next(data_iter)
                except StopIteration: data_iter = iter(self.dataloader); continue
                batch_data = batch_data.to(self.device, non_blocking=True)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                transfer_time = time.time() - transfer_start
                inference_start = time.time()
                with torch.no_grad(): _ = self.model(batch_data)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                inference_time = time.time() - inference_start
                self.batch_count += 1; self.inference_count += batch_data.size(0)
                self.total_transfer_time += transfer_time; self.total_inference_time += inference_time
            except Exception as e: print(f"Error: {e}"); break

def main():
    layer_descriptions = {
        4: ("Residual Head", ResidualHead),
        5: ("Squeeze-and-Excitation Head", SEHead),
        6: ("Transformer Encoder Head", TransformerHead),
        7: ("Non-Local Block Head", NonLocalHead),
        8: ("CBAM Head", CBAMHead),
        # 9: ("Spatial Transformer Head", SpatialTransformerHead),
    }
    parser = argparse.ArgumentParser(description='PyTorch Transfer Learning Head Testing')
    parser.add_argument('--layer', type=int, default=4, choices=layer_descriptions.keys(), help='Transfer learning head type.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference.')
    parser.add_argument('--dataset_size', type=int, default=100000, help='Total samples in dataset.')
    parser.add_argument('--duration', type=int, default=0, help='Duration to run in seconds (0 for indefinite).')
    parser.add_argument('--no_freeze', action='store_true', help='Do not freeze backbone.')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes.')
    parser.add_argument('--list_layers', action='store_true', help='List available heads and exit.')
    args = parser.parse_args()

    if args.list_layers:
        print("Available transfer learning head types:")
        for i, (desc, _) in sorted(layer_descriptions.items()):
            print(f"  --layer {i:<2}: {desc}")
        return

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    backbone = FeatureExtractor()
    head_name, HeadClass = layer_descriptions[args.layer]
    head = HeadClass(in_channels=backbone.output_channels, num_classes=args.num_classes)
    model = TransferLearningModel(backbone, head, freeze_backbone=not args.no_freeze).to(device)
    model.eval()

    print(f"\nModel Configuration:")
    print(f"  Head: {head_name}")
    print(f"  Backbone Frozen: {not args.no_freeze}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.get_trainable_params()
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    dataset = LargeRandomDataset(args.dataset_size, num_classes=args.num_classes)
    runner = StreamingInferenceRunner(model, device, dataset, batch_size=args.batch_size)
    
    def signal_handler(signum, frame):
        print("\nInterrupt signal received. Shutting down...")
        runner.stop_inference()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        runner.start_inference()
        if args.duration > 0:
            print(f"Running for {args.duration} seconds... Press Ctrl+C to stop early.")
            time.sleep(args.duration)
        else:
            print("Running indefinitely. Press Ctrl+C to stop.")
            runner.thread.join()
    finally:
        runner.stop_inference()

if __name__ == "__main__":
    main()
