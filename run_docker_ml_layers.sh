#!/bin/bash

# if [ -z "$1" ]; then
#   echo "Error: Please provide the number of samples as the first argument."
#   echo "Usage: $0 <num_samples>"
#   exit 1
# fi

LOGS_DIR="./logs_ml_layers"
mkdir -p $LOGS_DIR
make
num_samples_arg="3000000"

stop_all_containers() {
    sudo docker stop $(sudo docker ps -a -q) >/dev/null 2>/dev/null
}

run_test() {
    local background_program=$1
    
    echo "[*] Statistics $background_program"
    
    [ "$background_program" = "off" ] && nvidia-smi -L >/dev/null
    sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace --shm-size=16g -v ~/.cache/torch/:/root/.cache/torch -v ~/.cache/huggingface:/root/.cache/huggingface -w /workspace ai_image \
        ./pf_latency_llm -n $num_samples_arg -o ${LOGS_DIR}/${background_program}
        2>/dev/null | grep "stride"
}

start_background_model() {
    local num_layers=$1
    
    echo "[!] Starting $model in training mode"
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        python3 -u torch_layers.py --layer $num_layers &
    sleep 10
}

# Optional: Enable persistent mode for faster execution
# sudo nvidia-smi -pm 1

stop_all_containers

for i in {4..8}
do
    start_background_model "$i"
    run_test "$i"
    stop_all_containers
done

