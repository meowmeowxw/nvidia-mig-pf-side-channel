import cudf
import numpy as np
import time
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
import cupy as cp
import argparse
import json
from datetime import datetime
import signal
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='cuDF Continuous UVM Join Benchmark')
parser.add_argument('--workload', type=str, default='continuous_join', help='Workload name')
parser.add_argument('--duration', type=int, default=0, help='Duration in seconds (0 = infinite)')
parser.add_argument('--num-rows', type=int, default=30_000_000, help='Number of rows in main dataframe')
parser.add_argument('--join-rows', type=int, default=5_000_000, help='Number of rows in join dataframe')
parser.add_argument('--stats-interval', type=int, default=10, help='Print stats every N iterations')
args = parser.parse_args()

# Initialize RMM with UVM
rmm.reinitialize(
    pool_allocator=False,
    managed_memory=True,
    initial_pool_size=2**30,
    maximum_pool_size=2**35 
)
cp.cuda.set_allocator(rmm_cupy_allocator)

# Global variables for graceful shutdown
running = True
iteration_count = 0
timing_history = []

def signal_handler(sig, frame):
    global running
    print('\n\nSHUTDOWN_SIGNAL_RECEIVED')
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Configuration
NUM_ROWS = args.num_rows
JOIN_ROWS = args.join_rows
NUM_KEYS = NUM_ROWS // 100  # Key cardinality for joins

print("="*70)
print("CONTINUOUS UVM JOIN BENCHMARK")
print("="*70)
print(f"WORKLOAD:       {args.workload}")
print(f"MAIN_ROWS:      {NUM_ROWS:,}")
print(f"JOIN_ROWS:      {JOIN_ROWS:,}")
print(f"KEY_SPACE:      {NUM_KEYS:,}")
print(f"DURATION:       {'Infinite' if args.duration == 0 else f'{args.duration}s'}")
print(f"STATS_INTERVAL: Every {args.stats_interval} iterations")
print("="*70)
print("\nInitializing dataframes...")

# ============================================
# CREATE BASE DATAFRAMES (ONCE)
# ============================================
start_init = time.perf_counter()

# Main dataframe with multiple columns
df_main_data = {
    'join_key': np.random.randint(0, NUM_KEYS, NUM_ROWS),
    'value1': np.random.randn(NUM_ROWS).astype(np.float64),
    'value2': np.random.randn(NUM_ROWS).astype(np.float64),
    'value3': np.random.randint(0, 1000, NUM_ROWS, dtype=np.int64),
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], NUM_ROWS),
}
df_main = cudf.DataFrame(df_main_data)

# Join dataframe (smaller, recreated each iteration)
df_join_template = {
    'join_key': np.random.randint(0, NUM_KEYS, JOIN_ROWS),
    'join_value': np.random.randn(JOIN_ROWS).astype(np.float64),
    'join_factor': np.random.randn(JOIN_ROWS).astype(np.float64)
}

init_time = time.perf_counter() - start_init

print(f"Initialization complete in {init_time:.3f}s")
print(f"Main DF:  {df_main.shape} | {df_main.memory_usage(deep=True).sum() / (1024**3):.3f} GB")
print(f"Join DF template: {JOIN_ROWS:,} rows")
print("="*70)
print("\nStarting continuous join loop...")
print("Press Ctrl+C to stop gracefully\n")

# ============================================
# CONTINUOUS JOIN LOOP
# ============================================
start_time = time.perf_counter()
last_stats_time = start_time

try:
    while running:
        iteration_start = time.perf_counter()
        
        # Create new join dataframe with varying data (forces page migrations)
        df_join = cudf.DataFrame({
            'join_key': np.random.randint(0, NUM_KEYS, JOIN_ROWS),
            'join_value': np.random.randn(JOIN_ROWS).astype(np.float64),
            'join_factor': np.random.randn(JOIN_ROWS).astype(np.float64)
        })
        
        # Perform inner join (most memory-intensive operation)
        df_result = df_main.merge(df_join, on='join_key', how='inner')
        
        # Add some computation to ensure data is accessed
        result_sum = df_result['value1'].sum()
        result_mean = df_result['join_value'].mean()
        
        # Clean up result to trigger more page migrations
        del df_result
        del df_join
        
        iteration_time = time.perf_counter() - iteration_start
        timing_history.append(iteration_time)
        iteration_count += 1
        
        # Print stats at intervals
        if iteration_count % args.stats_interval == 0:
            elapsed = time.perf_counter() - start_time
            recent_timings = timing_history[-args.stats_interval:]
            avg_iter_time = np.mean(recent_timings)
            std_iter_time = np.std(recent_timings)
            min_iter_time = np.min(recent_timings)
            max_iter_time = np.max(recent_timings)
            iter_per_sec = args.stats_interval / (time.perf_counter() - last_stats_time)
            
            print(f"ITER {iteration_count:6d} | "
                  f"Elapsed: {elapsed:8.2f}s | "
                  f"Iter/s: {iter_per_sec:5.2f} | "
                  f"Avg: {avg_iter_time:6.3f}s | "
                  f"Std: {std_iter_time:6.3f}s | "
                  f"Min: {min_iter_time:6.3f}s | "
                  f"Max: {max_iter_time:6.3f}s")
            
            last_stats_time = time.perf_counter()
        
        # Check duration limit
        if args.duration > 0 and (time.perf_counter() - start_time) >= args.duration:
            print(f"\nDuration limit of {args.duration}s reached")
            running = False

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    running = False

# ============================================
# FINAL STATISTICS
# ============================================
total_elapsed = time.perf_counter() - start_time

print("\n" + "="*70)
print("BENCHMARK COMPLETE")
print("="*70)
print(f"Total iterations:     {iteration_count:,}")
print(f"Total time:           {total_elapsed:.3f}s")
print(f"Average iter time:    {np.mean(timing_history):.4f}s")
print(f"Std dev iter time:    {np.std(timing_history):.4f}s")
print(f"Min iter time:        {np.min(timing_history):.4f}s")
print(f"Max iter time:        {np.max(timing_history):.4f}s")
print(f"Iterations per sec:   {iteration_count / total_elapsed:.3f}")
print("="*70)

# Summary JSON
results = {
    'workload': args.workload,
    'timestamp': datetime.now().isoformat(),
    'config': {
        'num_rows': NUM_ROWS,
        'join_rows': JOIN_ROWS,
        'key_space': NUM_KEYS,
        'duration_seconds': args.duration,
    },
    'results': {
        'total_iterations': iteration_count,
        'total_time_seconds': total_elapsed,
        'iterations_per_second': iteration_count / total_elapsed,
        'avg_iteration_time': float(np.mean(timing_history)),
        'std_iteration_time': float(np.std(timing_history)),
        'min_iteration_time': float(np.min(timing_history)),
        'max_iteration_time': float(np.max(timing_history)),
    }
}

print("\nJSON_RESULTS_START")
print(json.dumps(results, indent=2))
print("JSON_RESULTS_END")

sys.exit(0)
