#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <unistd.h>
#include <time.h>
#include <cuda/atomic>


#define CHUNK0_SIZE (64L * 1024L * 1024L * 1024L * 1024L + 0x55554000000L)
#define CHUNK1_SIZE (41L * 1024L * 1024L * 1024L * 1024L + 0x0ffc8000000L)

uint8_t *chunk0 = 0;
uint8_t *chunk1 = 0;

#define L1_CACHE 106608
#define L2_CACHE (12 * 1024 * 1024)
#define CONSTANT_CACHE (64 * 1024)
#define CACHE_LINE 128

__global__ void l2_cache_cvwt(uint8_t *values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    uint64_t val = 0;
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val = __ldcv(&values[i]);
            val += i;
            __stwt(&values[i], val);
        }
    }
}

__global__ void l2_cache_atomicexchange(uint8_t *values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            __nv_atomic_exchange((int *)&values[i], (int *)&values[i + 8], (int *)&values[i + 16], __ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            // __stwt(&values[i], val);
        }
    }
}

__global__ void l2_cache_atomicExchAdd(uint8_t *values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    uint64_t val = 0;
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val = (uint64_t)atomicExch((unsigned long long *)&values[i], (unsigned long long)val);
            val += i;
            atomicAdd((unsigned long long *)&values[i], val);
        }
    }
}

__global__ void l2_cache_cgatomicExch(uint8_t *values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    uint64_t val = 0;
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val = __ldcg(&values[i]);
            val += i;
            atomicExch((unsigned long long *)&values[i], val);
        }
    }
}

__device__ __constant__ uint8_t c_values[CONSTANT_CACHE] = {0};

__global__ void l2_cache_constant_stcg(uint8_t *values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (CONSTANT_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (CONSTANT_CACHE / sizeof(uint8_t))) {
        end_idx = CONSTANT_CACHE / sizeof(uint8_t);
    }
    while (1) {
        uint64_t val = 0;
        for (int i = start_idx; i < end_idx; i += CONSTANT_CACHE) {
            val = __ldcg(&c_values[i]);
            val += i;
            __stcg(&values[i + 8], val);
        }
    }
}

__global__ void l2_cache_cawb(uint8_t *values, int num_threads) {

    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    uint64_t val = 0;
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val = __ldca(&values[i]);
            val += i;
            __stwb(&values[i], val);
        }
    }
}

__global__ void l2_cache_cvcg(uint8_t *values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    uint64_t val = 0;
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val = __ldcv(&values[i]);
            val += i;
            __stcg(&values[i], val);
        }
    }
}

__global__ void l2_cache_cvwb(uint8_t *values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    uint64_t val = 0;
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val = __ldcv(&values[i]);
            val += i;
            __stwb(&values[i], val);
        }
    }
}

__global__ void l2_cache_cgcg(uint8_t *d_values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    uint64_t val = 0;
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val = __ldcg(&d_values[i]);
            val += i;
            __stcg(&d_values[i], val);
        }
    }
}

__global__ void l1_cache(uint8_t *d_values, int num_threads) {
    int idx = threadIdx.x;
    int elements_per_thread = (L1_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L1_CACHE / sizeof(uint8_t))) {
        end_idx = L1_CACHE / sizeof(uint8_t);
    }
    printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    uint64_t val = 0;
    while (1) {
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val = __ldca(&d_values[i]);
            val += i;
            d_values[i] = val;
        }
    }
}

static inline __attribute__((always_inline)) uint64_t rdtscp(void) {
	uint64_t lo, hi;
	asm volatile("rdtscp\n" : "=a" (lo), "=d" (hi) :: "rcx");
	return (hi << 32) | lo;
}

static inline __attribute__((always_inline)) void mfence() {
	asm volatile ("mfence\n");
}

__global__ void page_fault_access(uint64_t *latency_value, uint64_t stride_size, uint64_t page) {
    volatile uint32_t x;
    uint64_t clk0 = 0;
    uint32_t idx = threadIdx.x;
    clk0 = clock64();
    uint64_t p = page + (idx * stride_size);
    asm volatile("ld.global.u32 %0, [%1];": "=r"(x) : "l"(p) : "memory");
    clk0 *= (x + 1);
    if (threadIdx.x == 0) {
        latency_value[threadIdx.x] = (clock64() - clk0);
    }
}

int main(int argc, char *argv[]) {
    cudaDeviceReset();
    uint8_t *d_values;
    uint8_t *dpin_values;
    uint8_t *h_values;
    uint64_t *d_latency_values, *gpu_latency_values, *host_latency_values;
    int num_threads = 256, iterations = 1000, stride_size = 0x1000;
    int option = 0;
    char *filepath = strdup("./logs_cache/l0");

    cudaMalloc(&d_values, L2_CACHE);
    cudaHostAlloc(&h_values, L2_CACHE, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&dpin_values, h_values, 0);
    cudaMallocManaged(&chunk0, CHUNK0_SIZE);
    cudaMallocManaged(&chunk1, CHUNK1_SIZE);

        static struct option long_options[] = {
        {"threads",    required_argument, 0, 't'},
        {"filepath",   required_argument, 0, 'f'},
        {"option",     required_argument, 0, 'o'},
        {"iterations", required_argument, 0, 'i'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int c;
    int option_index = 0;
    
    while ((c = getopt_long(argc, argv, "t:f:o:i:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 't':
                num_threads = atoi(optarg);
                if (num_threads <= 0) {
                    fprintf(stderr, "Error: Number of threads must be positive\n");
                    exit(1);
                }
                break;
            case 'f':
                free(filepath);
                filepath = strdup(optarg);
                break;
            case 'o':
                option = atoi(optarg);
                if (option < 0) {
                    fprintf(stderr, "Error: Option must be >= 0\n");
                    exit(1);
                }
                break;
            case 'i':
                iterations = atoi(optarg);
                if (iterations <= 0) {
                    fprintf(stderr, "Error: Number of iterations must be positive\n");
                    exit(1);
                }
                break;
            case '?':
                fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
                exit(1);
            default:
                abort();
        }
    }

    printf("Running with: threads=%d, filepath=%s, option=%d, iterations=%d\n", 
       num_threads, filepath, option, iterations);

    // parse arugment:
    // num_threads
    // filepath
    // option
    // iterations
    //
    uint64_t page = (uint64_t)chunk0;
    FILE *file_output;
    char buffer[32];
    int len;
    volatile uint64_t x;
    host_latency_values = (uint64_t *)malloc(sizeof(uint64_t) * iterations);
    uint32_t max_it = 100;
    int repeat = 1;
    if (iterations > max_it) {
        repeat = (uint32_t)(ceil(iterations / (float)max_it));
        iterations = max_it;
    }

    switch (option) {
        case 0:
            cudaMalloc(&d_latency_values, iterations * sizeof(uint64_t));
            file_output = fopen(filepath, "w");
            for (int j = 0; j < repeat; j++) {
                for (int i = 0; i < iterations; i++) {
                    page_fault_access<<<1,32>>>(&d_latency_values[i], stride_size, page);
                    cudaDeviceSynchronize();
                    uint64_t s = rdtscp();
                    x = *(uint64_t *)page;
                    mfence();
                    host_latency_values[i] = rdtscp() - s;
                    cudaDeviceSynchronize();
                    page = ((uint64_t)(chunk0)) + (0x10000 * (rand() % 30000));
                }
                gpu_latency_values = (uint64_t *)malloc(iterations * sizeof(uint64_t));
                cudaMemcpy(gpu_latency_values, d_latency_values, iterations * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                for (int i = 0; i < iterations; i++) {
                    len = snprintf(buffer, sizeof(buffer), "%lu, %lu\n", gpu_latency_values[i], host_latency_values[i]);
                    fwrite(buffer, 1, len, file_output);
                }
                cudaFree(chunk0);
                cudaFree(chunk1);
                usleep(1000);
                cudaMallocManaged(&chunk0, CHUNK0_SIZE);
                cudaMallocManaged(&chunk1, CHUNK1_SIZE);
            }
            fclose(file_output);
            free(gpu_latency_values);
            cudaFree(d_latency_values);
            break;
        case 1:
            l1_cache<<<1, 1>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 2:
            l2_cache_cgcg<<<8, num_threads>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 3:
            l2_cache_cvwt<<<8, num_threads>>>(dpin_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 4:
            l2_cache_cvcg<<<8, num_threads>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 5:
            l2_cache_cvwb<<<8, num_threads>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 6:
            l2_cache_cgatomicExch<<<8, num_threads>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 7:
            l2_cache_atomicexchange<<<8, num_threads>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 8:
            l2_cache_atomicExchAdd<<<8, num_threads>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 9:
            l2_cache_cawb<<<8, num_threads>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
        case 10:
            l2_cache_constant_stcg<<<8, num_threads>>>(d_values, num_threads);
            cudaDeviceSynchronize();
            break;
    }


    cudaFree(chunk0);
    cudaFree(chunk1);
    free(filepath);
}
