#!/bin/bash

LOGS_DIR="./logs_llm_pcie_monitor_noise"
mkdir -p $LOGS_DIR
make
num_samples_arg="400"

# Global variable to store the noiser container ID
NOISER_CONTAINER_ID=""

stop_all_containers() {
    sudo docker stop $(sudo docker ps -a -q) >/dev/null 2>/dev/null
}

stop_pcie_noise() {
    if [ -n "$NOISER_CONTAINER_ID" ]; then
        echo "[!] Stopping PCIe noise generator (Container: $NOISER_CONTAINER_ID)"
        sudo docker stop $NOISER_CONTAINER_ID >/dev/null 2>/dev/null
        NOISER_CONTAINER_ID=""
        sleep 1
    fi
}

run_test() {
    local background_program=$1
    
    echo "[*] Collecting PCIe throughput for $background_program"
    
    sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        ./pcie_monitor --num $num_samples_arg --output ${LOGS_DIR}/${background_program}
}

start_background_model() {
    local model=$1
    
    echo "[!] Starting $model in inference mode"
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        python3 -u vllm_runner.py --model $model &
    sleep 35
}

start_pcie_noise() {
    local noise_arg=$1
    local noise_label=$2
    
    # Stop any existing noise generator first
    stop_pcie_noise
    
    if [ "$noise_arg" = "none" ]; then
        echo "[!] No PCIe noise"
        return
    fi
    
    echo "[!] Starting PCIe noise generator: $noise_label (level $noise_arg)"
    NOISER_CONTAINER_ID=$(sudo docker run -d --rm --gpus "device=0:1" -v $PWD:/workspace -w /workspace ai_image \
        ./pcie_noiser $noise_arg)
    echo "[!] Noiser container ID: $NOISER_CONTAINER_ID"
    sleep 5
}

# Noise configurations: argument for pcie_noiser, label for output filename
NOISE_CONFIGS=(
    "none none"
    "0 low"
    "1 medium"
    "2 high"
)

# Test subset of models for noise robustness study
MODELS=(
    "Qwen2-1.5B"
    "GPT2-Medium"
    "GPT2-Large"
    "StableLM"
    "OLMo-1B"
    "OPT-125m"
    "Phi-3-Mini"
    "TinyLlama-1B"
    "GPT-Neo-2.7B"
)

echo "=== PCIe Monitor Noise Robustness Test ==="
echo "Output directory: $LOGS_DIR"
echo "Samples per test: $num_samples_arg"
echo "=========================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- Testing model: $model ---"
    stop_all_containers
    start_background_model "$model"
    
    for noise_config in "${NOISE_CONFIGS[@]}"; do
        noise_arg=$(echo $noise_config | cut -d' ' -f1)
        noise_label=$(echo $noise_config | cut -d' ' -f2)
        
        start_pcie_noise "$noise_arg" "$noise_label"
        run_test "${model}_${noise_label}"
        stop_pcie_noise
    done
done

stop_all_containers
echo ""
echo "[*] All tests complete. Logs in $LOGS_DIR"
echo "[*] Files created:"
ls -lh $LOGS_DIR/ | tail -n +2
