#!/bin/bash

LOGS_DIR="./logs_llm"
mkdir -p $LOGS_DIR
make
num_samples_arg="5000000"

stop_all_containers() {
    sudo docker stop $(sudo docker ps -a -q) >/dev/null 2>/dev/null
}

run_test() {
    local background_program=$1
    
    echo "[*] Statistics $background_program"
    
    [ "$background_program" = "off" ] && nvidia-smi -L >/dev/null
    sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        ./pf_latency_llm -n $num_samples_arg -o ${LOGS_DIR}/${background_program}
        2>/dev/null | grep "stride"
}

start_background_model() {
    local model=$1
    
    echo "[!] Starting $model in training mode"
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        python3 -u vllm_runner.py --model $model &
    sleep 40
}

# Optional: Enable persistent mode for faster execution
# sudo nvidia-smi -pm 1

stop_all_containers

MODELS=(
    "Starcoder2-3B"
    "Qwen2-1.5B"
    "Phi-3-Mini"
    "GPT2-Medium"
    "GPT2-Large"
    "GPT-Neo-2.7B"
    "OPT-125m"
    "TinyLlama-1B"
    "StableLM"
    "OLMo-1B"
)

echo "--- Run LLM ---"
for model in "${MODELS[@]}"; do
    stop_all_containers
    start_background_model "$model"
    run_test "$model"
done
stop_all_containers

    # "BLOOM-3B"
    # "T5-Small"
    # "RoBERTa-Base"
    # "ELECTRA-Small"
