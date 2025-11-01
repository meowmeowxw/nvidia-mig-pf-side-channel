# NVIDIA MIG Side Channel

## Introduction

[NVIDIA MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) is
a technology to partition a GPU into multiple GPU Instances (GIs). Each GI
has a slice of available computational (GPCs - SMs) and memory (VRAM/L2) resources.
MIG can be deployed on containers or virtual machine (through vGPU).
However, previous work has shown that the L3-TLB and PCIe bus are shared component
between GIs, enabling side channel to fingerprint workloads running on a co-located GI.
In this work we show that also the UVM (Unified Virtual Memory) page fault latency
can be used to fingerprint workloads running on co-located GIs.
We identified three main sources of contention that contribute to the page fault latency
variation:

1. PCIe bus usage.
2. Memory access operations (cache hints).
3. Shared kernel-level driver code and queue (fault buffer) - Only applicable to
concurrent UVM workloads in containerized environments.

## Tests

### Requirements and Setup

Tested on Ubuntu 24

```
./setup.sh
```

### Cross-GI Interference

```
./run_docker_cross_gi.sh
python3 plotter_cache.py -s p99 -o figs/ --log_scale on
```

### Side Channel on vLLM

```
./run_docker_vllm.sh
python3 classifier_llm.py --logs_dir ./logs_llm/ --window_size 30000 --step 3000
```
