#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include <libgen.h>
#include <nvml.h>

// Global variables for cleanup
nvmlDevice_t device;
int nvml_initialized = 0;
FILE *output_file = NULL;

void cleanup() {
    if (output_file) {
        fclose(output_file);
        output_file = NULL;
    }
    if (nvml_initialized) {
        nvmlShutdown();
        printf("\nNVML Shutdown.\n");
    }
}

void signal_handler(int sig) {
    printf("\nMonitoring stopped by user.\n");
    cleanup();
    exit(0);
}

int create_directory(const char *path) {
    char *path_copy = strdup(path);
    char *dir = dirname(path_copy);
    
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0755) == -1) {
            free(path_copy);
            return -1;
        }
    }
    
    free(path_copy);
    return 0;
}

void print_usage(const char *program_name) {
    printf("Usage: %s --output <file_path> --num <number_of_samples> [options]\n", program_name);
    printf("\nRequired arguments:\n");
    printf("  --output <path>     Output file path (e.g., ./data/throughput.csv)\n");
    printf("  --num <number>      Number of samples to collect\n");
    printf("\nOptional arguments:\n");
    printf("  --gpu <index>       GPU index to monitor (default: 0)\n");
    printf("  --interval <seconds> Polling interval in seconds (default: 0.5)\n");
    printf("  --help              Show this help message\n");
}

int main(int argc, char *argv[]) {
    char *output_path = NULL;
    int num_samples = 0;
    int gpu_index = 0;
    float interval = 0.5;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--num") == 0 && i + 1 < argc) {
            num_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            gpu_index = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
            interval = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (!output_path || num_samples <= 0) {
        printf("Error: --output and --num are required arguments\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (interval <= 0) {
        printf("Error: Interval must be greater than 0\n");
        return 1;
    }
    
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    
    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("NVML Error: Failed to initialize NVML: %s\n", nvmlErrorString(result));
        printf("Ensure NVIDIA drivers are installed and NVML library is accessible.\n");
        return 1;
    }
    nvml_initialized = 1;
    
    // Get device handle
    result = nvmlDeviceGetHandleByIndex(gpu_index, &device);
    if (result != NVML_SUCCESS) {
        printf("NVML Error: Failed to get device handle for GPU %d: %s\n", 
               gpu_index, nvmlErrorString(result));
        cleanup();
        return 1;
    }
    
    // Get device name
    char device_name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetName(device, device_name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result != NVML_SUCCESS) {
        printf("NVML Error: Failed to get device name: %s\n", nvmlErrorString(result));
        cleanup();
        return 1;
    }
    
    printf("Monitoring PCIe throughput for GPU %d: %s\n", gpu_index, device_name);
    printf("Collecting %d samples...\n", num_samples);
    printf("Output file: %s\n", output_path);
    printf("Press Ctrl+C to stop early.\n");
    printf("-------------------------------------------------------\n");
    
    // Create output directory if needed
    if (create_directory(output_path) != 0) {
        printf("Error: Failed to create output directory\n");
        cleanup();
        return 1;
    }
    
    // Open output file
    output_file = fopen(output_path, "w");
    if (!output_file) {
        printf("Error: Failed to open output file: %s\n", output_path);
        cleanup();
        return 1;
    }
    
    // Collect samples
    for (int sample = 0; sample < num_samples; sample++) {
        unsigned int rx_throughput, tx_throughput;
        
        // Get PCIe RX throughput (Host to GPU)
        result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &rx_throughput);
        if (result != NVML_SUCCESS) {
            printf("NVML Error: Failed to get RX throughput: %s\n", nvmlErrorString(result));
            cleanup();
            return 1;
        }
        
        // Get PCIe TX throughput (GPU to Host)
        result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &tx_throughput);
        if (result != NVML_SUCCESS) {
            printf("NVML Error: Failed to get TX throughput: %s\n", nvmlErrorString(result));
            cleanup();
            return 1;
        }
        
        // Write to file in format: rx, tx
        fprintf(output_file, "%u, %u\n", rx_throughput, tx_throughput);
        fflush(output_file);
        
        // Print progress
        time_t current_time = time(NULL);
        struct tm *tm_info = localtime(&current_time);
        char time_str[64];
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
        
        printf("Sample %d/%d - %s | RX: %7u KB/s | TX: %7u KB/s\n",
               sample + 1, num_samples, time_str, rx_throughput, tx_throughput);
        fflush(stdout);
        
        // Sleep before next sample (except for the last sample)
        // if (sample < num_samples - 1) {
        //     usleep((int)(interval * 1000000)); // Convert to microseconds
        // }
    }
    
    printf("\nData collection complete. %d samples written to %s\n", num_samples, output_path);
    
    cleanup();
    return 0;
}
