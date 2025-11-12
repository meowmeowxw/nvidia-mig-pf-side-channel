#!/bin/bash

LOGS_DIR="./logs_side_channel_cache_pcie"
mkdir -p $LOGS_DIR
make
num_samples_arg="400"

stop_all_containers() {
    sudo docker stop $(sudo docker ps -a -q) >/dev/null 2>/dev/null
}

run_test() {
    local background_program=$1
    
    echo "[*] Statistics $background_program"
    
    [ "$background_program" = "off" ] && nvidia-smi -L >/dev/null
    sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        ./pcie_monitor --num $num_samples_arg --output ${LOGS_DIR}/${background_program}
}

start_background_cache_accesses() {
    local cache_option=$1
    
    echo "[!] Starting $cache_option"
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        ./cache_accesses_loop --option $cache_option &
    sleep 4
}

# Optional: Enable persistent mode for faster execution
# sudo nvidia-smi -pm 1

stop_all_containers

OPTIONS=(
    # "3"
    "9"
    "10"
    "2"
    "4"
    "8"
    "7"
    "6"
)

echo "--- Run GPU kernels cache ---"
for option in "${OPTIONS[@]}"; do
    stop_all_containers
    start_background_cache_accesses "$option"
    run_test "$option"
done
stop_all_containers

