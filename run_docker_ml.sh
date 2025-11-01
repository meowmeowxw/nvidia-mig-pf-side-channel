#!/bin/sh

LOGS_DIR="./logs_ml"
mkdir -p $LOGS_DIR
make
num_samples_arg="2000000"

stop_all_containers() {
    sudo docker stop $(sudo docker ps -a -q) >/dev/null 2>/dev/null
}

run_test() {
    local background_program=$1
    
    echo "[*] Statistics $background_program"
    
    [ "$background_program" = "off" ] && nvidia-smi -L >/dev/null
    sudo docker run --rm --gpus "device=0:0" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        ./pf_latency -n $num_samples_arg -o ${LOGS_DIR}/${background_program}
        2>/dev/null | grep "stride"
}

start_background_model() {
    local model=$1
    
    echo "[!] Starting $model in training mode"
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        python3 -u torch_ml.py $model --mode train &
    sleep 10
}

# Optional: Enable persistent mode for faster execution
# sudo nvidia-smi -pm 1

stop_all_containers

start_background_model "mobilenetv2"
run_test "mobilenetv2"
stop_all_containers

start_background_model "resnet"
run_test "resnet"
stop_all_containers

start_background_model "vgg"
run_test "vgg"
stop_all_containers

start_background_model "densenet"
run_test "densenet"
stop_all_containers
