NVCC = nvcc
FLAGS = -arch=sm_80 -diag-suppress 2464 -lcurand

FILES = attack.cu cache_accesses_victim.cu pf_latency_llm.cu

BINS = $(FILES:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean
