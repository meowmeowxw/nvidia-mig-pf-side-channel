#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict

# --- Helper functions (copied from your previous script) ---
def read_raw_data(filepath: str, columns: List[int] = [0, 1]) -> Dict[int, np.ndarray]:
    """Read raw float values from a file for specified columns."""
    raw_data = {col: [] for col in columns}
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or ',' not in line:
                continue
            try:
                parts = line.strip().split(', ')
                for col in columns:
                    if col < len(parts):
                        raw_data[col].append(float(parts[col]))
                    else:
                        raw_data[col].append(0.0)
            except (ValueError, IndexError):
                continue
    for col in columns:
        raw_data[col] = np.array(raw_data[col])
    return raw_data

def extract_features(raw_data: Dict[int, np.ndarray], columns: List[int], window_size: int, step: int) -> List[List[float]]:
    """Extract features from raw data with specified window and step."""
    if step == 0:
        step = window_size
    features = []
    min_length = min(len(raw_data[col]) for col in columns if col in raw_data)
    if min_length < window_size:
        return []
    for i in range(0, min_length - window_size + 1, step):
        # We only need the mean for this visualization
        mean_val = np.mean(raw_data[columns[0]][i:i+window_size])
        features.append(mean_val)
    return features

def get_feature_distribution_data(logs_dir: str, models_to_plot: List[str], window_size: int) -> pd.DataFrame:
    """Generates a Pandas DataFrame with feature values for plotting."""
    plot_data = []
    for model in models_to_plot:
        model_files = [f for f in os.listdir(logs_dir) if f.startswith(model + '_')]
        
        # Combine data from all noise levels for each model
        combined_raw_data = {0: []}
        for fname in sorted(model_files):
            fpath = os.path.join(logs_dir, fname)
            raw_data = read_raw_data(fpath, columns=[0])
            combined_raw_data[0].extend(raw_data[0])
        
        combined_raw_data[0] = np.array(combined_raw_data[0])
        
        # Extract features (just the mean)
        # Use a small step to get more data points for a smoother distribution
        features = extract_features(combined_raw_data, columns=[0], window_size=window_size, step=int(window_size/2))

        for feat_val in features:
            plot_data.append({'Model': model, 'Feature Value': feat_val})
            
    return pd.DataFrame(plot_data)


def main():
    parser = argparse.ArgumentParser(description='Visualize feature distributions for different window sizes.')
    parser.add_argument("--logs_dir", required=True, help="Directory containing the LLM log files.")
    args = parser.parse_args()

    # --- Configuration ---
    MODELS_TO_PLOT = ['GPT2-Large', 'Qwen2-1.5B', 'Starcoder2-3B']
    FEATURE_NAME = 'Mean of PCIe Read Throughput (col0_mean)'
    
    # Window sizes to compare
    window_sizes = [50, 150] 

    # Create the plot
    fig, axes = plt.subplots(1, len(window_sizes), figsize=(8 * len(window_sizes), 6), sharey=True)
    fig.suptitle('Feature Distribution by Window Size', fontsize=16)

    for i, size in enumerate(window_sizes):
        print(f"Generating data for window size: {size}...")
        df = get_feature_distribution_data(args.logs_dir, MODELS_TO_PLOT, window_size=size)
        
        if df.empty:
            print(f"Warning: Not enough data to generate features for window size {size}.")
            continue

        ax = axes[i]
        sns.kdeplot(data=df, x='Feature Value', hue='Model', fill=True, common_norm=False, ax=ax, palette="viridis")
        
        ax.set_title(f'Window Size = {size}')
        ax.set_xlabel(FEATURE_NAME)
        if i > 0:
            ax.set_ylabel('') # Remove redundant Y-axis label

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('feature_distribution_comparison.png')
    print("\nPlot saved as feature_distribution_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
