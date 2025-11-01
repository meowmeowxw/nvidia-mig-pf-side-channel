NVCC = nvcc
FLAGS = -arch=sm_80 -diag-suppress 2464 -lcurand

FILES = attack.cu cache_accesses_victim.cu pf_latency_llm.cu pf_latency.cu cache_accesses_loop.cu victim.cu

ifndef CUDA_PATH
	CUDA_PATH := /usr/local/cuda
endif
NVIDIA_DRIVER_PATH=/home/ubuntu/tools/NVIDIA-Linux-x86_64-570.133.07/
INCLUDES := -I${CUDA_PATH}/include
INCLUDES += -I${NVIDIA_DRIVER_PATH}/kernel-open/common/inc
INCLUDES += -I${NVIDIA_DRIVER_PATH}/kernel-open/nvidia
INCLUDES += -I${NVIDIA_DRIVER_PATH}/kernel-open/nvidia-uvm

BINS = $(FILES:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@


pcie_monitor: pcie_monitor.c
	gcc -o pcie_monitor pcie_monitor.c $(INCLUDES) -lnvidia-ml

clean:
	rm -f $(BINS)

.PHONY: all clean
