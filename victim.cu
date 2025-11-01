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

#define WAIT_TIME 2000000000L
#define ITERATIONS 50000

uint64_t *chunk0 = 0;
uint64_t *chunk1 = 0;

enum kernel_option {
    WAIT,
    COMPUTE_SQRT,
    COMPUTE_SQRT_CU_MEMORY,
    COMPUTE_SQRT_UVM,
    COMPUTE_SQRT_CONSTANT_MEMORY,
    COMPUTE_SQRT_STACK,
    COMPUTE_SQRT_GLOBAL,
    COMPUTE_LOG
};

char *get_str_kernel_option(enum kernel_option cmd) {
    switch(cmd) {
        case WAIT: return "WAIT";
        case COMPUTE_SQRT: return "COMPUTE_SQRT";
        case COMPUTE_SQRT_CU_MEMORY: return "COMPUTE_SQRT_CU_MEMORY";
        case COMPUTE_SQRT_UVM: return "COMPUTE_SQRT_UVM";
        case COMPUTE_SQRT_CONSTANT_MEMORY: return "COMPUTE_SQRT_CONSTANT_MEMORY";
        case COMPUTE_SQRT_STACK: return "COMPUTE_SQRT_STACK";
    }
    return "UNKNOWN";
}

__global__ void _compute_sqrt(uint64_t *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = clock64(), end = 0;

    float value = 1.0f + (idx % 17) * 0.1f;
    for (uint64_t i = 0; i < 0x1000; i++) {
        value = sqrtf(value + 0.01f);
    }

    if (idx == 0) {
        start *= (int)value;
        end = clock64() - start;
        output[idx] = end;
    }
}

__device__ __constant__ float c_values[0x100] = {
0.76, 0.77, 0.6, 0.21, 0.33, 0.72, 0.31, 0.28, 0.67, 0.84, 0.37, 0.35, 0.73, 0.48, 0.66, 0.18, 0.74, 0.31, 0.55, 0.19, 0.4, 0.69, 0.21, 0.31, 0.71, 0.43, 0.71, 0.55, 0.25, 0.86, 0.19, 0.54, 0.56, 0.23, 0.76, 0.85, 0.46, 0.68, 0.23, 0.1, 0.79, 0.17, 0.55, 0.53, 0.13, 0.27, 0.21, 0.53, 0.49, 0.12, 0.67, 0.69, 0.12, 0.3, 0.43, 0.72, 0.12, 0.39, 0.54, 0.26, 0.67, 0.6, 0.58, 0.58, 0.17, 0.18, 0.34, 0.33, 0.82, 0.72, 0.75, 0.25, 0.75, 0.61, 0.45, 0.81, 0.48, 0.75, 0.36, 0.37, 0.23, 0.73, 0.37, 0.9, 0.38, 0.85, 0.47, 0.79, 0.6, 0.53, 0.68, 0.8, 0.47, 0.61, 0.7, 0.53, 0.34, 0.43, 0.7, 0.46, 0.67, 0.54, 0.36, 0.52, 0.62, 0.35, 0.26, 0.19, 0.57, 0.82, 0.34, 0.21, 0.48, 0.16, 0.75, 0.72, 0.34, 0.68, 0.2, 0.45, 0.7, 0.82, 0.46, 0.5, 0.42, 0.4, 0.83, 0.37, 0.31, 0.23, 0.43, 0.72, 0.83, 0.63, 0.68, 0.54, 0.7, 0.71, 0.38, 0.6, 0.7, 0.39, 0.65, 0.64, 0.14, 0.25, 0.36, 0.51, 0.13, 0.21, 0.56, 0.17, 0.78, 0.77, 0.57, 0.57, 0.11, 0.51, 0.23, 0.69, 0.72, 0.46, 0.41, 0.71, 0.59, 0.36, 0.59, 0.47, 0.84, 0.44, 0.4, 0.5, 0.66, 0.44, 0.89, 0.38, 0.35, 0.82, 0.23, 0.7, 0.84, 0.12, 0.55, 0.22, 0.29, 0.53, 0.6, 0.2, 0.48, 0.68, 0.43, 0.3, 0.83, 0.35, 0.16, 0.85, 0.44, 0.25, 0.81, 0.24, 0.88, 0.18, 0.49, 0.83, 0.21, 0.24, 0.51, 0.59, 0.12, 0.64, 0.27, 0.51, 0.81, 0.69, 0.88, 0.26, 0.58, 0.59, 0.83, 0.82, 0.62, 0.66, 0.56, 0.85, 0.3, 0.26, 0.66, 0.8, 0.86, 0.31, 0.4, 0.73, 0.46, 0.44, 0.22, 0.29, 0.66, 0.54, 0.87, 0.79, 0.31, 0.58, 0.14, 0.3, 0.57, 0.15, 0.31, 0.81, 0.81, 0.82, 0.69, 0.66, 0.41, 0.88, 0.76, 0.68
};

__global__ void _compute_sqrt_constant_memory(uint64_t *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = clock64(), end = 0;

    float value = 1.0f + (idx % 17) * 0.1f;
    for (uint64_t i = 0; i < 0x100; i++) {
        value = sqrtf(c_values[i % 0x100] + value);
    }

    start *= (int)value;
    end = clock64() - start;
    output[idx] = end;
}

// values allocated with cudaMallocManaged()
__global__ void _compute_sqrt_params(uint64_t *output, float *values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = clock64(), end = 0;

    float value = 1.0f + (idx % 17) * 0.1f;
    for (uint64_t i = 0; i < 0x1000; i++) {
        value = sqrtf(values[i % 0x100] + value);
    }

    start *= (int)value;
    end = clock64() - start;
    output[idx] = end;
}


__global__ void _compute_sqrt_global(uint64_t *output, float *values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = clock64(), end = 0;
    float value = 1.0f + (idx % 17) * 0.1f;
    for (uint64_t i = 0; i < 0x10000; i++) {
        value = sqrtf(values[i] + value);
    }
    start *= (int)value;
    end = clock64() - start;
    output[idx] = end;
}

__global__ void _compute_sqrt_stack(uint64_t *output) {
    float stack_values[0x100] = {
    0.76, 0.77, 0.6, 0.21, 0.33, 0.72, 0.31, 0.28, 0.67, 0.84, 0.37, 0.35, 0.73, 0.48, 0.66, 0.18, 0.74, 0.31, 0.55, 0.19, 0.4, 0.69, 0.21, 0.31, 0.71, 0.43, 0.71, 0.55, 0.25, 0.86, 0.19, 0.54, 0.56, 0.23, 0.76, 0.85, 0.46, 0.68, 0.23, 0.1, 0.79, 0.17, 0.55, 0.53, 0.13, 0.27, 0.21, 0.53, 0.49, 0.12, 0.67, 0.69, 0.12, 0.3, 0.43, 0.72, 0.12, 0.39, 0.54, 0.26, 0.67, 0.6, 0.58, 0.58, 0.17, 0.18, 0.34, 0.33, 0.82, 0.72, 0.75, 0.25, 0.75, 0.61, 0.45, 0.81, 0.48, 0.75, 0.36, 0.37, 0.23, 0.73, 0.37, 0.9, 0.38, 0.85, 0.47, 0.79, 0.6, 0.53, 0.68, 0.8, 0.47, 0.61, 0.7, 0.53, 0.34, 0.43, 0.7, 0.46, 0.67, 0.54, 0.36, 0.52, 0.62, 0.35, 0.26, 0.19, 0.57, 0.82, 0.34, 0.21, 0.48, 0.16, 0.75, 0.72, 0.34, 0.68, 0.2, 0.45, 0.7, 0.82, 0.46, 0.5, 0.42, 0.4, 0.83, 0.37, 0.31, 0.23, 0.43, 0.72, 0.83, 0.63, 0.68, 0.54, 0.7, 0.71, 0.38, 0.6, 0.7, 0.39, 0.65, 0.64, 0.14, 0.25, 0.36, 0.51, 0.13, 0.21, 0.56, 0.17, 0.78, 0.77, 0.57, 0.57, 0.11, 0.51, 0.23, 0.69, 0.72, 0.46, 0.41, 0.71, 0.59, 0.36, 0.59, 0.47, 0.84, 0.44, 0.4, 0.5, 0.66, 0.44, 0.89, 0.38, 0.35, 0.82, 0.23, 0.7, 0.84, 0.12, 0.55, 0.22, 0.29, 0.53, 0.6, 0.2, 0.48, 0.68, 0.43, 0.3, 0.83, 0.35, 0.16, 0.85, 0.44, 0.25, 0.81, 0.24, 0.88, 0.18, 0.49, 0.83, 0.21, 0.24, 0.51, 0.59, 0.12, 0.64, 0.27, 0.51, 0.81, 0.69, 0.88, 0.26, 0.58, 0.59, 0.83, 0.82, 0.62, 0.66, 0.56, 0.85, 0.3, 0.26, 0.66, 0.8, 0.86, 0.31, 0.4, 0.73, 0.46, 0.44, 0.22, 0.29, 0.66, 0.54, 0.87, 0.79, 0.31, 0.58, 0.14, 0.3, 0.57, 0.15, 0.31, 0.81, 0.81, 0.82, 0.69, 0.66, 0.41, 0.88, 0.76, 0.68
    };
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = clock64(), end = 0;
    float value = 1.0f + (idx % 17) * 0.1f;
    for (uint64_t i = 0; i < 0x1000; i++) {
        value = sqrtf(stack_values[i % 0x100] + value * 0.002f);
    }
    start *= (int)value;
    end = clock64() - start;
    output[idx] = end;
}

void dispatch(enum kernel_option cmd, uint32_t iterations, char *workload) {
    float h_values[0x100] = {
    0.76, 0.77, 0.6, 0.21, 0.33, 0.72, 0.31, 0.28, 0.67, 0.84, 0.37, 0.35, 0.73, 0.48, 0.66, 0.18, 0.74, 0.31, 0.55, 0.19, 0.4, 0.69, 0.21, 0.31, 0.71, 0.43, 0.71, 0.55, 0.25, 0.86, 0.19, 0.54, 0.56, 0.23, 0.76, 0.85, 0.46, 0.68, 0.23, 0.1, 0.79, 0.17, 0.55, 0.53, 0.13, 0.27, 0.21, 0.53, 0.49, 0.12, 0.67, 0.69, 0.12, 0.3, 0.43, 0.72, 0.12, 0.39, 0.54, 0.26, 0.67, 0.6, 0.58, 0.58, 0.17, 0.18, 0.34, 0.33, 0.82, 0.72, 0.75, 0.25, 0.75, 0.61, 0.45, 0.81, 0.48, 0.75, 0.36, 0.37, 0.23, 0.73, 0.37, 0.9, 0.38, 0.85, 0.47, 0.79, 0.6, 0.53, 0.68, 0.8, 0.47, 0.61, 0.7, 0.53, 0.34, 0.43, 0.7, 0.46, 0.67, 0.54, 0.36, 0.52, 0.62, 0.35, 0.26, 0.19, 0.57, 0.82, 0.34, 0.21, 0.48, 0.16, 0.75, 0.72, 0.34, 0.68, 0.2, 0.45, 0.7, 0.82, 0.46, 0.5, 0.42, 0.4, 0.83, 0.37, 0.31, 0.23, 0.43, 0.72, 0.83, 0.63, 0.68, 0.54, 0.7, 0.71, 0.38, 0.6, 0.7, 0.39, 0.65, 0.64, 0.14, 0.25, 0.36, 0.51, 0.13, 0.21, 0.56, 0.17, 0.78, 0.77, 0.57, 0.57, 0.11, 0.51, 0.23, 0.69, 0.72, 0.46, 0.41, 0.71, 0.59, 0.36, 0.59, 0.47, 0.84, 0.44, 0.4, 0.5, 0.66, 0.44, 0.89, 0.38, 0.35, 0.82, 0.23, 0.7, 0.84, 0.12, 0.55, 0.22, 0.29, 0.53, 0.6, 0.2, 0.48, 0.68, 0.43, 0.3, 0.83, 0.35, 0.16, 0.85, 0.44, 0.25, 0.81, 0.24, 0.88, 0.18, 0.49, 0.83, 0.21, 0.24, 0.51, 0.59, 0.12, 0.64, 0.27, 0.51, 0.81, 0.69, 0.88, 0.26, 0.58, 0.59, 0.83, 0.82, 0.62, 0.66, 0.56, 0.85, 0.3, 0.26, 0.66, 0.8, 0.86, 0.31, 0.4, 0.73, 0.46, 0.44, 0.22, 0.29, 0.66, 0.54, 0.87, 0.79, 0.31, 0.58, 0.14, 0.3, 0.57, 0.15, 0.31, 0.81, 0.81, 0.82, 0.69, 0.66, 0.41, 0.88, 0.76, 0.68, 
    };
    float *d_values;
    float *uvm_values;
    float *g_values;
    if (cudaMalloc(&d_values, sizeof(h_values)) != cudaSuccess) {
        printf("error\n");
        return;
    };
    if (cudaMallocManaged(&uvm_values, sizeof(h_values)) != cudaSuccess) {
        printf("error\n");
        return;
    }
    if (cudaMalloc(&g_values, 0x10000 * sizeof(float)) != cudaSuccess) {
        printf("error\n");
        return;
    }
    cudaMemset(g_values, 1, 0x10000 * sizeof(float));
    memcpy(uvm_values, &h_values, sizeof(h_values));
    cudaMemAdvise(&uvm_values, sizeof(h_values), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemcpy(d_values, &h_values, sizeof(h_values), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    uint64_t *latencies = (uint64_t *)malloc(iterations * sizeof(uint64_t));
    if (!latencies) {
        fprintf(stderr, "malloc error\n");
        return;
    }
    uint64_t *output;
    if (cudaMalloc(&output, 0x64) != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged error\n");
        return;
    }
    char *cmd_str = get_str_kernel_option(cmd);
    for (int i = 0; i < iterations; i++) {
        switch (cmd) {
            case COMPUTE_SQRT:
                _compute_sqrt<<<1,1>>>(output);
                break;
            case COMPUTE_SQRT_CU_MEMORY:
                _compute_sqrt_params<<<1,1>>>(output, d_values);
                break;
            case COMPUTE_SQRT_GLOBAL:
                _compute_sqrt_global<<<1,1>>>(output, g_values);
                break;
            case COMPUTE_SQRT_UVM:
                _compute_sqrt_params<<<1,1>>>(output, uvm_values);
                break;
            case COMPUTE_SQRT_CONSTANT_MEMORY:
                _compute_sqrt_constant_memory<<<1,1>>>(output);
                break;
            case COMPUTE_SQRT_STACK:
                _compute_sqrt_stack<<<1,1>>>(output);
                break;
            default:
                break;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(&latencies[i], output, 8, cudaMemcpyDeviceToHost);
        printf("%s, %s, 0x%lx\n", cmd_str, workload, (uint64_t)latencies[i]);
        if (cmd == COMPUTE_SQRT_UVM) {
            uvm_values[0] = uvm_values[1];
        }
        fflush(stdout);
        // sleep(1);
    }
    cudaFree(output);
    cudaFree(d_values);
    cudaFree(g_values);
    cudaFree(uvm_values);
    free(latencies);
}

int main(int argc, char *argv[]) {

    cudaDeviceReset();

    enum kernel_option kopt = COMPUTE_SQRT;
    uint32_t iterations = 100;
    char *workload = strdup("unknown");

    int c;

    while (1) {
        static struct option long_options[] = {
            {"kernel_option",    required_argument, 0, 'o'},
            {"iterations", required_argument, 0, 'i'},
            {"workload", required_argument, 0, 'w'}
        };

        int option_index = 0;
        c = getopt_long(argc, argv, "vos:t:n:i:w:", long_options, &option_index);

        if (c == -1)
            break;

        switch (c) {
            case 'o':
                if (strcmp(optarg, "wait") == 0) {
                    kopt = WAIT;
                } else if (strcmp(optarg, "compute_sqrt") == 0) {
                    kopt = COMPUTE_SQRT;
                } else if (strcmp(optarg, "compute_sqrt_cu_memory") == 0) {
                    kopt = COMPUTE_SQRT_CU_MEMORY;
                } else if (strcmp(optarg, "compute_sqrt_uvm") == 0) {
                    kopt = COMPUTE_SQRT_UVM;
                } else if (strcmp(optarg, "compute_sqrt_constant_memory") == 0) {
                    kopt = COMPUTE_SQRT_CONSTANT_MEMORY;
                } else if (strcmp(optarg, "compute_sqrt_global") == 0) {
                    kopt = COMPUTE_SQRT_GLOBAL;
                } else if (strcmp(optarg, "compute_sqrt_stack") == 0) {
                    kopt = COMPUTE_SQRT_STACK;
                } else if (strcmp(optarg, "compute_log") == 0) {
                    kopt = COMPUTE_LOG;
                } else {
                    fprintf(stderr, "Invalid option value: %s\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'i':
                iterations = atoi(optarg);
                break;
            case 'w':
                workload = strdup(optarg);
                break;
            case '?':
                exit(EXIT_FAILURE);
            default:
                abort();
        }
    }

    dispatch(kopt, iterations, workload);
    cudaDeviceSynchronize();

    free(workload);
    cudaFree(chunk0);
    cudaFree(chunk1);

}

