#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>

__global__ void dummy_kernel(uint8_t *data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (uint8_t)((idx * 7 + blockIdx.x) & 0xFF);
    }
}

typedef enum {
    NOISE_LOW,
    NOISE_MEDIUM, 
    NOISE_HIGH,
} NoiseLevel;

typedef struct {
    size_t min_size;
    size_t max_size;
    int min_interval_us;
    int max_interval_us;
    const char* name;
} NoiseConfig;

NoiseConfig get_noise_config(NoiseLevel level) {
    NoiseConfig configs[] = {
        // LOW: 1KB-1 MB, 10-20ms intervals
        {1*1024, 1*1024*1024, 10000, 20000, "low"},
        // MEDIUM: 1-10 MB, 5-10ms intervals  
        {1*1024*1024, 10*1024*1024, 5000, 10000, "medium"},
        // HIGH: 10-50 MB, 1-5ms intervals
        {10*1024*1024, 50*1024*1024, 1000, 5000, "high"},
    };
    return configs[level];
}

size_t random_range(size_t min, size_t max) {
    return min + (rand() % (max - min + 1));
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <noise_level: 0=low, 1=medium, 2=high, 3=extreme> [duration_seconds]\n", argv[0]);
        fprintf(stderr, "  Or: %s none (no noise, just blocks)\n", argv[0]);
        return 1;
    }

    // Special case: "none" mode just blocks forever (for baseline tests)
    if (strcmp(argv[1], "none") == 0) {
        fprintf(stderr, "[PCIe Noise] NONE mode - blocking indefinitely\n");
        while(1) sleep(3600);
        return 0;
    }

    NoiseLevel level = (NoiseLevel)atoi(argv[1]);
    int duration_sec = (argc >= 3) ? atoi(argv[2]) : 0; // 0 = infinite
    
    NoiseConfig config = get_noise_config(level);
    
    // Allocate maximum possible buffer size
    size_t max_buffer = config.max_size;
    uint8_t *h_data, *d_data;
    
    cudaMallocHost(&h_data, max_buffer);
    cudaMalloc(&d_data, max_buffer);
    
    // Initialize with random data
    srand(time(NULL));
    for (size_t i = 0; i < max_buffer; i++) {
        h_data[i] = (uint8_t)(rand() & 0xFF);
    }
    
    fprintf(stderr, "[PCIe Noise] Level: %s\n", config.name);
    fprintf(stderr, "[PCIe Noise] Transfer size: %zu MB - %zu MB\n", 
            config.min_size/(1024*1024), config.max_size/(1024*1024));
    fprintf(stderr, "[PCIe Noise] Interval: %d - %d us\n",
            config.min_interval_us, config.max_interval_us);
    fprintf(stderr, "[PCIe Noise] Duration: %s\n", 
            duration_sec > 0 ? "limited" : "infinite");
    
    time_t start_time = time(NULL);
    unsigned long iterations = 0;
    size_t total_bytes = 0;
    
    while (1) {
        // Random transfer size for this iteration
        size_t transfer_size = random_range(config.min_size, config.max_size);
        
        // H2D transfer
        cudaMemcpy(d_data, h_data, transfer_size, cudaMemcpyHostToDevice);
        
        // Random kernel work (50% chance to skip for more variability)
        if (rand() % 2 == 0) {
            int blocks = (transfer_size / (256 * 1024)) + 1;
            dummy_kernel<<<blocks, 256>>>(d_data, transfer_size);
        }
        
        // D2H transfer (50% chance to skip - sometimes only H2D)
        if (rand() % 2 == 0) {
            cudaMemcpy(h_data, d_data, transfer_size, cudaMemcpyDeviceToHost);
        }
        
        cudaDeviceSynchronize();
        
        total_bytes += transfer_size;
        iterations++;
        
        // Random interval
        int interval = random_range(config.min_interval_us, config.max_interval_us);
        usleep(interval);
        
        // Check duration
        if (duration_sec > 0 && (time(NULL) - start_time) >= duration_sec) {
            break;
        }
        
        // Print stats every 1000 iterations
        if (iterations % 1000 == 0) {
            double elapsed = difftime(time(NULL), start_time);
            double throughput_gbps = (total_bytes / elapsed) / (1024.0*1024.0*1024.0);
            fprintf(stderr, "[PCIe Noise] Iterations: %lu, Avg throughput: %.2f GB/s\n",
                    iterations, throughput_gbps);
        }
    }
    
    double elapsed = difftime(time(NULL), start_time);
    double throughput_gbps = (total_bytes / elapsed) / (1024.0*1024.0*1024.0);
    fprintf(stderr, "[PCIe Noise] Completed: %lu iterations in %.1f seconds\n", 
            iterations, elapsed);
    fprintf(stderr, "[PCIe Noise] Average throughput: %.2f GB/s\n", throughput_gbps);
    
    cudaFreeHost(h_data);
    cudaFree(d_data);
    return 0;
}
