#include <bits/types/FILE.h>
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
#include <math.h>
#include <time.h>

// Values taken from the TunneLs for Bootlegging paper
#define CHUNK0_SIZE (64L * 1024L * 1024L * 1024L * 1024L + 0x55554000000L)
#define CHUNK1_SIZE (41L * 1024L * 1024L * 1024L * 1024L + 0x0ffc8000000L)

uint8_t *chunk0 = 0;
uint8_t *chunk1 = 0;

static inline __attribute__((always_inline)) uint64_t rdtscp(void) {
	uint64_t lo, hi;
	asm volatile("rdtscp\n" : "=a" (lo), "=d" (hi) :: "rcx");
	return (hi << 32) | lo;
}

static inline __attribute__((always_inline)) void mfence() {
	asm volatile ("mfence\n");
}

__global__ void put(uint64_t *page, uint64_t x1) {
    page[0] = x1;
    page[1] = x1;
    page[2] = x1;
}

// We call this kernel with 1 warp (32 threads). With stride_size = 65536 (64KB),
// which is >= typical page size, each thread targets a different memory page,
// thus intending to cause an individual page fault per thread on first access.
__global__ void kernel_memory_access(uint64_t *latency_value, uint64_t stride_size, uint64_t page) {
    volatile uint32_t x;
    uint64_t clk0 = 0;
    uint32_t idx = threadIdx.x;
    clk0 = clock64();
    uint64_t p = page + (idx * stride_size);
    // printf("p: 0x%lx\n", (uint64_t)p);
    asm volatile("ld.global.u32 %0, [%1];": "=r"(x) : "l"(p) : "memory");
    // Do not let the compiler optimize the load away and keep the
    // order of the operation. x will always be 0 (in theory)
    clk0 *= (x + 1);
    if (threadIdx.x == 0) {
        latency_value[threadIdx.x] = (clock64() - clk0);
    }
}

int attack(uint32_t iterations, FILE *file_output, uint64_t *values,
        uint64_t *h2d, uint64_t *d2h, uint64_t page) {
    uint32_t num_sms = 1;
    uint32_t num_threads = 32;
    uint32_t stride_size = (64L * 1024L);
    
    // uint64_t page = 0x700000000000;
    uint64_t original_page = page;
    volatile uint64_t dummy = 0;

    srand(time(NULL));

    for (int i = 0; i < iterations; i++) {
        *(uint64_t *)page = 0;
    	cudaDeviceSynchronize();
        kernel_memory_access<<<num_sms, num_threads>>>(&values[i], stride_size, page);
        cudaDeviceSynchronize();

        uint64_t s = rdtscp();
        dummy = *(uint64_t *)page;
        mfence();
        d2h[i] = rdtscp() - s;

        page = original_page + (0x10000 * (rand() % 10000));
        // printf("page: 0x%lx\n", page);
        cudaDeviceSynchronize();
        // sleep(10);
    }
    // printf("iterations done\n");
    
    cudaMemcpy(h2d, values, sizeof(uint64_t) * iterations, cudaMemcpyDeviceToHost);
    for (int i = 0; i < iterations; i++) {
        char buffer[36];
        int len = snprintf(buffer, sizeof(buffer), "%lu, %lu\n", h2d[i], d2h[i]); // , d2h[i]);
        fwrite(buffer, 1, len, file_output);
        // fprintf(stderr, "%ld\n", res[i]);
    }
    fflush(file_output);
    return 0;
}

int main(int argc, char *argv[]) {

    cudaDeviceReset();

    // hoard a large address space, make sure that ASLR is disabled
    // chunk0 = 0x100000000000 --> 0x500000000000
    cudaMallocManaged(&chunk0, CHUNK0_SIZE);
    // chunk1 = 0x580000000000 --> 0x800000000000
    cudaMallocManaged(&chunk1, CHUNK1_SIZE);

    uint32_t num_samples = 1;
    uint32_t iterations = 100;
    uint32_t repeat = 1;
    char *output = strdup("./output.txt");

    int c;

    while (1) {
        static struct option long_options[] = {
            {"num_samples", required_argument, 0, 'n'},
            {"output", required_argument, 0, 'o'},
            {0, 0, 0, 0}
        };

        int option_index = 0;
        c = getopt_long(argc, argv, "n:o:i:", long_options, &option_index);

        if (c == -1)
            break;

        switch (c) {
            case 'o':
                free(output);
                output = strdup(optarg);
                break;
            case 'n':
                num_samples = atoi(optarg);
                break;
            case '?':
                exit(EXIT_FAILURE);
            default:
                abort();
        }
    }

    iterations = num_samples;
    uint32_t max_it = 2000;
    if (num_samples > max_it) {
        repeat = (uint32_t)(ceil(num_samples / (float)max_it));
        iterations = max_it;
    }

    fprintf(stderr, "[*] repeat: %u, iterations: %u, num_samples: %u\n", repeat, iterations, repeat * iterations);
    FILE *file_output = fopen(output, "w");
    uint64_t *values = NULL, *h2d = NULL, *d2h = NULL;
    if (cudaMalloc(&values, sizeof(uint64_t) * iterations) != cudaSuccess) {
        fprintf(stderr, "error values\n");
        goto done;
    }
    h2d = (uint64_t *)malloc(sizeof(uint64_t) * iterations);
    if (h2d == NULL) {
        fprintf(stderr, "error h2d\n");
        cudaFree(values);
        goto done;
    }
    d2h = (uint64_t *)malloc(sizeof(uint64_t) * iterations);
    if (d2h == NULL) {
        fprintf(stderr, "error d2h\n");
        free(h2d);
        cudaFree(values);
        goto done;
    }

    for (int j = 0; j < repeat; j++) {
        if (attack(iterations, file_output, values, h2d, d2h, (uint64_t)chunk0) < 0) {
            printf("ERROR\n");
            goto done;
        };
        cudaFree(chunk0);
        cudaFree(chunk1);
        usleep(1000);
        cudaMallocManaged(&chunk0, CHUNK0_SIZE);
        cudaMallocManaged(&chunk1, CHUNK1_SIZE);
        // printf("chunk0: %p, chunk1: %p\n", chunk0, chunk1);
    }

    cudaDeviceSynchronize();

done:
    cudaFree(chunk0);
    cudaFree(chunk1);
    free(output);
}

