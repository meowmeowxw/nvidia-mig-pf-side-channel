#!/bin/sh

LOGS_DIR="./logs_simple"
mkdir -p $LOGS_DIR
make
num_samples_arg=120000

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
        python3 -u torch_models.py $model --mode train &
    sleep 10
}

start_background_vllm() {
    local model=$1
    echo "[!] Starting $model"
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        python3 -u vllm_runner.py --model $model &
    sleep 60
}

start_background_cudf() {
    sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
        python3 -u ./cudf_example.py &
    sleep 10
}

start_background_program() {
        local kernel_option=$1
        sudo docker run --rm --gpus "device=0:1" -v $PWD:/workspace --shm-size=16g -w /workspace ai_image \
            ./cache_accesses_loop --option $kernel_option &
        sleep 2
}

# Optional: Enable persistent mode for faster execution
# sudo nvidia-smi -pm 1

stop_all_containers
run_test "inactive"
stop_all_containers

# # 
start_background_program "2"
run_test "ld_cg_st_cg"
stop_all_containers

start_background_model "mobilenetv2"
run_test "pytorch"
stop_all_containers

start_background_vllm "Phi-3-Mini"
run_test "vLLM"
stop_all_containers

start_background_cudf
run_test "cudf"
stop_all_containers

