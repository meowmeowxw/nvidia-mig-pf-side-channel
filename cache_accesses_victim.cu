#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <unistd.h>
#include <time.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#define CHUNK0_SIZE (64L * 1024L * 1024L * 1024L * 1024L + 0x55554000000L)
#define CHUNK1_SIZE (41L * 1024L * 1024L * 1024L * 1024L + 0x0ffc8000000L)

uint8_t *chunk0 = 0;
uint8_t *chunk1 = 0;

#define L1_CACHE (128 * 64)
#define L2_CACHE (128 * 256)
#define CACHE_LINE 128

__global__ void l2_cache_cvwt(uint8_t *values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += __ldcv(&values[i]);
            __stwt(&values[i + 8], val);
        }
        if (idx == 0) {
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}

__global__ void l2_cache_atomicexchange(uint8_t *values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            __nv_atomic_exchange((int *)&values[i], (int *)&values[i + 8], (int *)&values[i + 16], __ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            val += values[i];
        }
        if (idx == 0) {
            // printf("val: %ld\n", val);
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}

__global__ void l2_cache_atomicExchAdd(uint8_t *values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += (uint64_t)atomicExch((unsigned long long *)&values[i], (unsigned long long)val);
            atomicAdd((unsigned long long *)&values[i + 8], val);
        }
        if (idx == 0) {
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}

__global__ void l2_cache_cgatomicExch(uint8_t *values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += __ldcg(&values[i]);
            atomicExch((unsigned long long *)&values[i + 8], val);
        }
        if (idx == 0) {
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}

__global__ void l2_cache_cawb(uint8_t *values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += __ldca(&values[i]);
            __stwb(&values[i + 8], val);
        }
        if (idx == 0) {
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}


__device__ __constant__ uint8_t c_values[L2_CACHE] = {0};

__global__ void l2_cache_constant_stcg(uint8_t *values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += __ldcg(&c_values[i]);
            __stcg(&values[i + 8], val);
        }
        if (idx == 0) {
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}

__global__ void l2_cache_cvcg(uint8_t *values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += __ldcv(&values[i]);
            __stcg(&values[i + 8], val);
        }
        if (idx == 0) {
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}

#define UVM_SIZE 16777216
#define UVM_LINE 65536

__device__ int simple_random_range(int a, int b, unsigned int seed) {
    if (a > b) {
        int temp = a;
        a = b;
        b = temp;
    }
    unsigned int rng = seed;
    rng = rng * 1664525u + 1013904223u;
    int range = b - a + 1;
    return a + (rng % range);
}

__device__ int random_range(curandState* state, int a, int b) {
    if (a > b) {
        // Swap if a > b
        int temp = a;
        a = b;
        b = temp;
    }
    
    int range = b - a + 1;  // +1 to make it inclusive
    return a + (curand(state) % range);
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void uvm_load(uint8_t *values, int num_threads, uint64_t *output, int iterations, uint8_t *page) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    uint64_t start_idx = idx * elements_per_thread;
    uint64_t end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // uint32_t seed = clock();
    uint64_t start = clock64(), end = 0, val = 0;
    // int i = random_range(state, 0, 214748369);
    // printf("i: %d\n", i);
    // int i = simple_random_range(0, 214748369, seed);
    val += __ldcg(page);
    start *= (val + 1);
    end = clock64() - start;
    *output = end;
}

__global__ void l2_cache_cvwb(uint8_t *values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += __ldcv(&values[i]);
            __stwb(&values[i + 8], val);
        }
        if (idx == 0) {
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}

__global__ void l2_cache_cgcg(uint8_t *d_values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L2_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L2_CACHE / sizeof(uint8_t))) {
        end_idx = L2_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += __ldcg(&d_values[i]);
            __stcg(&d_values[i + 8], val);
        }
        if (idx == 0) {
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
        }
    }
}

__global__ void l1_cache(uint8_t *d_values, int num_threads, uint64_t *output, int iterations) {
    int idx = threadIdx.x;
    int elements_per_thread = (L1_CACHE / sizeof(uint8_t)) / num_threads;
    int start_idx = idx * elements_per_thread;
    int end_idx = start_idx + elements_per_thread;
    
    if (end_idx > (L1_CACHE / sizeof(uint8_t))) {
        end_idx = L1_CACHE / sizeof(uint8_t);
    }
    // printf("[%d] start_idx: %d, end_idx: %d\n", idx, start_idx, end_idx);
    
    for (int j = 0; j < iterations; j++) {
        uint64_t start = clock64(), end = 0, val = 0;
        for (int i = start_idx; i < end_idx; i += CACHE_LINE) {
            val += __ldca(&d_values[i]);
            d_values[i + 8] = val;
        }
        if (idx == 0) {
            // printf("val: %ld\n", val);
            start *= (val + 1);
            end = clock64() - start;
            output[j] = end;
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

char *get_option(int option) {
    switch (option) {
        case 0: return "none";
        case 1: return "l1";
        case 2: return "ld.cg + st.cg";
        case 3: return "ld.cv + st.wt";
        case 4: return "ld.cv + st.cg";
        case 5: return "ld.cv + st.wb";
        case 6: return "ld.cg + atomicExch";
        case 7: return "nv_atomic_exchange";
        case 8: return "atomicExch + atomicAdd";
        case 9: return "ld.ca + st.wb";
        case 10: return "ld.cg (constant) + st.cg";
        case 11: return "UVM ld.cg";
        default: return "error";
    }
}
int main(int argc, char *argv[]) {
    cudaDeviceReset();
    uint8_t *d_values;
    uint8_t *dpin_values;
    uint8_t *h_values;
    uint64_t *latency_values;
    uint64_t *d_latency_values, *gpu_latency_values, *host_latency_values;
    int num_threads = 1, iterations = 1000, stride_size = 0x1000;
    int option = 0;
    char *filepath = strdup("./logs_cache/l0");
    char *workload = strdup("test");

    cudaMalloc(&d_values, L2_CACHE);
    cudaHostAlloc(&h_values, L2_CACHE, cudaHostAllocMapped);
    memset(h_values, 0, L2_CACHE);
    cudaHostGetDevicePointer(&dpin_values, h_values, 0);
    cudaMallocManaged(&chunk0, CHUNK0_SIZE);
    cudaMallocManaged(&chunk1, CHUNK1_SIZE);

        static struct option long_options[] = {
        {"threads",    required_argument, 0, 't'},
        {"filepath",   required_argument, 0, 'f'},
        {"option",     required_argument, 0, 'o'},
        {"iterations", required_argument, 0, 'i'},
        {"workload",   required_argument, 0, 'w'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int c;
    int option_index = 0;
    
    while ((c = getopt_long(argc, argv, "t:f:o:i:h:w", long_options, &option_index)) != -1) {
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
            case 'w':
                free(workload);
                workload = strdup(optarg);
                break;
            case '?':
                fprintf(stderr, "Try '%s --help' for more information.\n", argv[0]);
                exit(1);
            default:
                abort();
        }
    }

    // printf("Running with: threads=%d, filepath=%s, option=%d, iterations=%d\n", 
    //    num_threads, filepath, option, iterations);

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
    cudaMalloc(&d_latency_values, sizeof(uint64_t) * iterations);
    if (option == 0) {
        if (iterations > max_it) {
            repeat = (uint32_t)(ceil(iterations / (float)max_it));
            iterations = max_it;
        }
    }
    latency_values = (uint64_t *)malloc(sizeof(uint64_t) * iterations);

    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    setup_kernel<<<1,1>>>(d_state, time(NULL));
    cudaDeviceSynchronize();

    switch (option) {
        case 0:
            // cudaMalloc(&d_latency_values, iterations * sizeof(uint64_t));
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
            break;
        case 1:
            l1_cache<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 2:
            l2_cache_cgcg<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 3:
            l2_cache_cvwt<<<1, num_threads>>>(dpin_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 4:
            l2_cache_cvcg<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 5:
            l2_cache_cvwb<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 6:
            l2_cache_cgatomicExch<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 7:
            l2_cache_atomicexchange<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 8:
            l2_cache_atomicExchAdd<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 9:
            l2_cache_cawb<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 10:
            l2_cache_constant_stcg<<<1, num_threads>>>(d_values, num_threads, d_latency_values, iterations);
            cudaDeviceSynchronize();
            break;
        case 11:
            srand(time(NULL));
            for (int i = 0; i < iterations; i++) {
                uint64_t page = ((uint64_t)chunk0) + 0x40000 * (rand() % 8000);
                volatile uint8_t x = *(uint8_t *)page;
                mfence();
                uvm_load<<<1, num_threads>>>(chunk0, num_threads, &d_latency_values[i], iterations, (uint8_t *)page);
                cudaDeviceSynchronize();
            }
            // cudaDeviceSynchronize();
            // setup_kernel<<<1,1>>>(d_state, time(NULL) * time(NULL));
            // cudaDeviceSynchronize();
            // uvm_load<<<1, num_threads>>>(chunk1, num_threads, &d_latency_values[iterations / 2], iterations / 2, d_state);
            // cudaDeviceSynchronize();
            break;
    }

    cudaMemcpy(latency_values, d_latency_values, sizeof(uint64_t) * iterations, cudaMemcpyDeviceToHost);
    cudaFree(d_latency_values);

    for (int i = 0; i < iterations; i++) {
        printf("%s, %s, %ld\n", get_option(option), workload, latency_values[i]);
    }

    free(workload);
    cudaFree(chunk0);
    cudaFree(chunk1);
    cudaFree(d_state);
    free(filepath);
}
