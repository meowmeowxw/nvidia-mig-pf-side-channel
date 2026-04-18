#!/usr/bin/env python3

import argparse
import time
import os
from pathlib import Path
from vllm import LLM, SamplingParams

def check_model_downloaded(model_path):
    if os.path.exists(model_path) and os.path.isdir(model_path):
        model_files = list(Path(model_path).glob("*.bin")) + \
                     list(Path(model_path).glob("*.safetensors")) + \
                     list(Path(model_path).glob("config.json"))
        return len(model_files) > 0
    return False

def download_model(model_id, local_path):
    print(f"[*] Downloading model: {model_id} to {local_path}")
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False
        )
        print(f"[✓] Successfully downloaded: {model_id}")
        return True
    except Exception as e:
        print(f"[!] Failed to download {model_id}: {e}")
        return False

def download_all_models(models_dict, models_dir):
    print(f"[*] Checking and downloading models to: {models_dir}")
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, model_id in models_dict.items():
        local_path = os.path.join(models_dir, model_name)
        
        if check_model_downloaded(local_path):
            print(f"[✓] Model already downloaded: {model_name}")
        else:
            download_model(model_id, local_path)
    print("[*] Download process completed!")

def main():
    parser = argparse.ArgumentParser(description="vLLM Continuous Inference Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="GPT2-Medium",
        help="The name of the model to run (e.g., 'Qwen2-1.5B')."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all models before running inference."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory to store downloaded models (default: ./models)."
    )
    args = parser.parse_args()

    vllm_supported_models = {
        "Qwen2-1.5B": "Qwen/Qwen2-1.5B",
        "Phi-3-Mini": "microsoft/Phi-3-mini-4k-instruct",
        "GPT2-Medium": "gpt2-medium",
        "Starcoder2-3B": "bigcode/starcoder2-3b",
        "GPT2-Large": "gpt2-large",
        "GPT-Neo-2.7B": "EleutherAI/gpt-neo-2.7B",
        "BLOOM-3B": "bigscience/bloom-3b",
        "OPT-125m": "facebook/opt-125m",
        "TinyLlama-1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "StableLM": "stabilityai/stablelm-2-1_6b-chat",
        "Phi-4-Mini": "microsoft/Phi-4-mini-instruct",
        "OLMo-1B": "allenai/OLMo-1B-hf",
    }

    if args.download:
        download_all_models(vllm_supported_models, args.models_dir)
        return

    local_model_path = os.path.join(args.models_dir, args.model)
    if check_model_downloaded(local_model_path):
        print(f"[*] Using locally downloaded model from: {local_model_path}")
        model_id = local_model_path
    else:
        model_id = vllm_supported_models.get(args.model)
        if not model_id:
            print(f"Model '{args.model}' is not supported or not found in the list.")
            return
        print(f"[*] Using HuggingFace model: {model_id}")

    print(f"[*] Starting inference for model: {model_id}")

    try:
        if args.model == "Phi-4-Mini":
            llm = LLM(model=model_id, quantization="awq")
        else:
            llm = LLM(model=model_id)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=200
        )
        
        while True:
            prompts = [
                "Hello, my name is",
                "The story of our universe begins with",
                "Once upon a time, in a land far, far away",
                "The future of artificial intelligence is"
            ]
            
            outputs = llm.generate(prompts, sampling_params)
            # for output in outputs:
            #     prompt = output.prompt
            #     generated_text = output.outputs[0].text
            #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    except KeyboardInterrupt:
        print("\n[*] Stopping continuous inference.")
    except Exception as e:
        print(f"\n[!] An error occurred: {e}")

if __name__ == "__main__":
    main()
