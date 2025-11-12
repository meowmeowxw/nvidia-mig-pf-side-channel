#!/bin/bash

LOGS_DIR="./logs_llm_pcie_noise"
mkdir -p $LOGS_DIR
make
num_samples_arg="5000000"

stop_all_containers() {
    sudo docker stop $(sudo docker ps -a -q) >/dev/null 2>/dev/null
}

run_test() {
    local background_program=$1
    
    echo "[*] Statistics $background_program"
    
    sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        ./pf_latency_llm -n $num_samples_arg -o ${LOGS_DIR}/${background_program} \
        2>/dev/null | grep "stride"
}

start_background_model() {
    local model=$1
    
    echo "[!] Starting $model in inference mode"
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        python3 -u vllm_runner.py --model $model &
    sleep 40
}

start_pcie_noise() {
    local noise_arg=$1
    local noise_label=$2
    
    if [ "$noise_arg" = "none" ]; then
        echo "[!] No PCIe noise"
        return
    fi
    
    echo "[!] Starting PCIe noise generator: $noise_label"
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace -w /workspace ai_image \
        ./pcie_noiser $noise_arg &
    sleep 5
}

# Noise configurations: argument for pcie_noiser, label for output filename
NOISE_CONFIGS=(
    "0 low"
    "1 medium"
    "2 high"
    "none none"
)

# Test subset of models for noise robustness study
MODELS=(
    "Qwen2-1.5B"
    "GPT2-Medium"
    "GPT2-Large"
    "StableLM"
    "OLMo-1B"
    # "OPT-125m"
    # "Phi-3-Mini"
    # "TinyLlama-1B"
    # "GPT-Neo-2.7B"
)

echo "=== PCIe Noise Robustness Test ==="
for model in "${MODELS[@]}"; do
    for noise_config in "${NOISE_CONFIGS[@]}"; do
        noise_arg=$(echo $noise_config | cut -d' ' -f1)
        noise_label=$(echo $noise_config | cut -d' ' -f2)
        
        stop_all_containers
        start_background_model "$model"
        start_pcie_noise "$noise_arg" "$noise_label"
        run_test "${model}_${noise_label}"
    done
done

stop_all_containers
echo "[*] All tests complete. Logs in $LOGS_DIR"
