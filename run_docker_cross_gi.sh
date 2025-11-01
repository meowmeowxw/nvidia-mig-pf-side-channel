#!/bin/bash

LOGS_DIR="./logs_cache"      # e.g., ./logs_cache

# Configuration
ITERATIONS=4002
STRIDE="0x1000"
DOCKER_IMAGE="ai_image"

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"
make
killall cache_accesses_victim
killall attack

stop_all_containers() {
    sudo docker stop $(sudo docker ps -a -q) >/dev/null 2>/dev/null
}

# Function to run attack and victim with specified parameters
run_attack_victim() {
    local num_blocks=$1
    local num_threads=$2
    local option=$3
    local workload_name="interrupt-storm-${num_blocks}x${num_threads}"
    local log_file="${LOGS_DIR}/${option}-${workload_name}"
    
    echo "Running workload: $workload_name"
    echo "Log file: $log_file"
    
    # Start attack in background using Docker
    sudo docker run --rm --gpus "device=0:1" \
        -v $PWD:/workspace -w /workspace \
        --shm-size=16g -d \
        --name attack_container \
        "$DOCKER_IMAGE" ./attack \
        --num_threads "$num_threads" \
        --num_blocks "$num_blocks" \
        --stride 0x1000 > /dev/null 2>&1
    
    local attack_container="attack_container"
    echo "Attack started in container: $attack_container"
    sleep 2
    
    # Run victim and wait for completion using Docker
    sudo docker run --rm --gpus "device=0:0" \
        -v $PWD:/workspace -w /workspace \
        --shm-size=16g \
        "$DOCKER_IMAGE" ./cache_accesses_victim \
        --iterations "$ITERATIONS" \
        --option "$option" \
        --workload "$workload_name" > "$log_file"
    
    local victim_exit_code=$?
    
    # Kill the attack container
    stop_all_containers
    
    if [ $victim_exit_code -eq 0 ]; then
        echo "Workload $workload_name completed successfully"
    else
        echo "Workload $workload_name failed with exit code: $victim_exit_code"
    fi
    
    echo "---"
}

# Main execution
echo "Starting CUDA attack-victim benchmark with Docker"
echo "Victim device: GPU 0:0"
echo "Attack device: GPU 0:1"
echo "Iterations: $ITERATIONS"
echo "Stride: $STRIDE"
echo ""
stop_all_containers

MIN_OPTION=2
MAX_OPTION=11

for i in $(seq $MIN_OPTION $MAX_OPTION)
do
    sudo docker run --rm --gpus "device=0:1" \
        -v $PWD:/workspace -w /workspace \
        --shm-size=16g -d \
        "$DOCKER_IMAGE" ./attack --copy 1 > /dev/null 2>&1
    sleep 2

    sudo docker run --rm --gpus "device=0:0" \
        -v $PWD:/workspace -w /workspace \
        --shm-size=16g \
        "$DOCKER_IMAGE" ./cache_accesses_victim \
        --iterations "$ITERATIONS" \
        --option "$i" \
        --workload copy_data > "${LOGS_DIR}/${i}-copy_data"
    stop_all_containers
done

for i in $(seq $MIN_OPTION $MAX_OPTION)
do
    run_attack_victim 1 1 $i
    run_attack_victim 1 32 $i
    run_attack_victim 1 64 $i
    run_attack_victim 1 256 $i
    run_attack_victim 8 512 $i
    run_attack_victim 8 1024 $i
    run_attack_victim 32 1024 $i
    run_attack_victim 256 1024 $i
done

for i in $(seq $MIN_OPTION $MAX_OPTION)
do
    echo "Running PyTorch"
    sudo docker run --rm --gpus "device=0:1" \
        -v $PWD:/workspace -v ~/.cache/huggingface:/root/.cache/huggingface \
        -w /workspace --shm-size=16g -d \
        "$DOCKER_IMAGE" python3 -u torch_ml.py --mode train --batch-size 32 vgg > /dev/null 2>&1
    sleep 6

    sudo docker run --rm --gpus "device=0:0" \
        -v $PWD:/workspace -w /workspace \
        --shm-size=16g \
        "$DOCKER_IMAGE" ./cache_accesses_victim \
        --iterations "$ITERATIONS" \
        --option "$i" \
        --workload pytorch > "${LOGS_DIR}/${i}-pytorch"
    stop_all_containers

    # Run baseline (no attack)
    echo "Running baseline (no attack)"
    sudo docker run --rm --gpus "device=0:0" \
        -v $PWD:/workspace -w /workspace \
        --shm-size=16g \
        "$DOCKER_IMAGE" ./cache_accesses_victim \
        --iterations "$ITERATIONS" \
        --option "$i" \
        --workload inactive > "${LOGS_DIR}/${i}-inactive"

    if [ $? -eq 0 ]; then
        echo "Baseline completed successfully"
    else
        echo "Baseline failed"
    fi
    echo "---"
    stop_all_containers
done

