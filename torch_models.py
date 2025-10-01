import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import sys
import argparse

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(batch_size=32, input_size=224, train=True):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)), # Resize images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download or load CIFAR10: {e}")
        print("Please check your internet connection and permissions.")
        sys.exit(1)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    mode = "Training" if train else "Test"
    print(f"CIFAR-10 dataset loaded. {mode} set size: {len(dataset)}")
    return dataloader

def get_model(model_name: str, num_classes: int = 10, pretrained=False):
    model = None
    in_features = 0
    input_size = 224 # Default input size for most models

    # Use pretrained weights if requested
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    
    if model_name == 'resnet':
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'vgg':
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        # VGG's classifier is a Sequential module, modify the last layer
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == 'densenet':
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'mobilenetv2':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        # MobileNetV2's classifier is also Sequential
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        print(f"Error: Model '{model_name}' is not supported.")
        print("Supported models: resnet, vgg, densenet, mobilenetv2")
        sys.exit(1)

    pretrained_status = "pretrained" if pretrained else "untrained"
    print(f"Loaded {pretrained_status} model: {model_name}")
    return model, input_size

def train(model, device, trainloader, epochs=1):
    """Trains the model on the provided data loader."""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nStarting training for {epochs} epoch(s)...")
    start_time = time.time()
    total_batches = len(trainloader)

    for epoch in range(epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0 or (i + 1) == total_batches:
                 print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{total_batches}, Loss: {running_loss / 100:.4f}')
                 running_loss = 0.0

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_duration:.2f} seconds.")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    end_time = time.time()
    print(f'\nTraining completed in {end_time - start_time:.2f} seconds')

def classify(model, device, dataloader):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    print("\nStarting classification...")
    start_time = time.time()

    total_batches = len(dataloader)
    dataset_size = len(dataloader.dataset)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress = int(50 * batch_idx / total_batches)
            percent = 100 * total / dataset_size
            sys.stdout.write('\r')
            sys.stdout.write(f"[{'=' * progress}{' ' * (50 - progress)}] {percent:.1f}% ({total}/{dataset_size}) images classified")
            sys.stdout.flush()
            
    accuracy = 100 * correct / total
    end_time = time.time()
    print(f'\nClassification completed in {end_time - start_time:.2f} seconds')
    print(f'Accuracy on the dataset: {accuracy:.2f}%')

def benchmark(model_name: str, epochs_to_run: int, mode='train', batch_size=32):
    device = get_device()
    print(f'Using device: {device}, batch_size: {batch_size}')
    
    pretrained = (mode == 'classify')
    model, input_size = get_model(model_name, num_classes=10, pretrained=pretrained)
    
    dataloader = load_data(batch_size=batch_size, input_size=input_size, train=True)
    
    if mode == 'train':
        train(model, device, dataloader, epochs=epochs_to_run)
    elif mode == 'classify':
        # We would like to constantly classify 
        for _ in range(epochs_to_run):
            classify(model, device, dataloader)

def parse_arguments():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description='PyTorch model benchmarking tool for CIFAR-10',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'model',
        type=str,
        choices=['resnet', 'vgg', 'densenet', 'mobilenetv2'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'classify'],
        default='train',
        help='Mode to run: train the model or classify with pretrained weights'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for data loading'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000000,
        help='Number of epochs to run'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    
    benchmark(args.model, args.epochs, args.mode, batch_size=args.batch_size)
