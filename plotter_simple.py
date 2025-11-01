#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatterSciNotation, FuncFormatter, FormatStrFormatter
import seaborn as sns
from scipy.stats import gmean

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper")
sns.set_palette("husl")

def parse_file(filepath, column=0):
    """Parse a log file and return the specified column."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments
                try:
                    values = line.split(', ')
                    if len(values) > column:
                        data.append(float(values[column]))
                except (ValueError, IndexError):
                    continue  # Skip malformed lines
    return np.array(data)

def compute_windowed_statistics(data, window_size, num_windows=None):
    """Compute statistics over sliding windows."""
    medians = []
    means = []
    stds = []
    q25 = []
    q75 = []
    p90 = []
    p95 = []
    tail_avgs = []  # Top 20% average
    
    end = len(data) if num_windows is None else num_windows * window_size
    for i in range(0, end, window_size):
        window = data[i:i+window_size]
        if len(window) > 0:
            medians.append(np.median(window))
            means.append(np.mean(window))
            stds.append(np.std(window))
            q25.append(np.percentile(window, 25))
            q75.append(np.percentile(window, 75))
            p90.append(np.percentile(window, 90))
            p95.append(np.percentile(window, 95))
            
            # Compute tail average (mean of top 20%)
            sorted_window = np.sort(window)
            top_20_percent_idx = int(0.8 * len(sorted_window))
            if top_20_percent_idx < len(sorted_window):
                tail_avg = np.mean(sorted_window[top_20_percent_idx:])
            else:
                tail_avg = sorted_window[-1]  # If window too small, use max
            tail_avgs.append(tail_avg)
    
    return {
        'medians': np.array(medians),
        'means': np.array(means),
        'stds': np.array(stds),
        'q25': np.array(q25),
        'q75': np.array(q75),
        'p90': np.array(p90),
        'p95': np.array(p95),
        'tail_avgs': np.array(tail_avgs)
    }

def should_use_log_scale(all_y_values, threshold_factor=10):
    """
    Determine if log scale should be used based on value separation.
    
    Args:
        all_y_values: List of arrays containing y-values for all series
        threshold_factor: Minimum ratio between max and min to trigger log scale
    
    Returns:
        bool: True if log scale should be used
    """
    # Flatten all values
    all_vals = np.concatenate([vals for vals in all_y_values if len(vals) > 0])
    
    # Remove zeros and negative values as they can't be plotted on log scale
    positive_vals = all_vals[all_vals > 0]
    
    if len(positive_vals) == 0:
        return False
    
    min_val = np.min(positive_vals)
    max_val = np.max(positive_vals)
    
    # Check if the range spans multiple orders of magnitude
    if min_val > 0:
        ratio = max_val / min_val
        return ratio >= threshold_factor
    
    return False

def get_publication_colors(n_colors):
    """Get a set of distinct, colorblind-friendly colors for publication."""
    if n_colors <= 10:
        colors = [ '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',]
        colors = [ '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf',]

        # Use a predefined colorblind-friendly palette
        # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        #          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return colors[:n_colors]
    else:
        # Use seaborn's husl palette for more colors
        return sns.color_palette("husl", n_colors)

def format_filename_for_label(filename):
    if filename == "ld_cg_st_cg":
        return "ld.cg + st.cg"
    if filename == "vLLM":
        return "vLLM"
    if filename == "pytorch":
        return "PyTorch"
    if filename == "cudf":
        return "cuDF"
    return filename.title()
    # name = filename.replace('.log', '').replace('.txt', '').replace('.csv', '')
    # name = name.replace('_', ' ')
    # return name

def main():
    parser = argparse.ArgumentParser(description='Plot windowed statistics from log files')
    parser.add_argument('--logs_dir', required=True, help='Directory containing log files')
    parser.add_argument('--window_size', type=int, default=2000, help='Window size for statistics computation')
    parser.add_argument('--num_windows', type=int, default=None, help='Number of windows')
    parser.add_argument('--column', type=int, default=0, help='Column to use (0-indexed)')
    parser.add_argument('--fig_output', required=True, help='Output figure path (PDF recommended)')
    parser.add_argument('--plot_type', choices=['median', 'mean', 'p90', 'p95', 'tail_avg', 'std'], default='median', 
                       help='Type of statistic to plot (tail_avg = mean of top 20%)')
    parser.add_argument('--show_uncertainty', action='store_true', 
                       help='Show uncertainty bands (std for mean, IQR for median)')
    parser.add_argument('--title', default='', help='Custom plot title')
    parser.add_argument('--xlabel', default='Window Index', help='X-axis label')
    parser.add_argument('--ylabel', default='Cycles', help='Y-axis label')
    parser.add_argument('--figsize', nargs=2, type=float, default=[8, 6], 
                       help='Figure size in inches (width height)')
    parser.add_argument('--log_scale', choices=['auto', 'on', 'off'], default='auto',
                       help='Y-axis log scale: auto (detect), on (force), off (disable)')
    parser.add_argument('--log_threshold', type=float, default=10.0,
                       help='Minimum max/min ratio to trigger auto log scale (default: 10)')
    
    args = parser.parse_args()
    
    # Ensure output is PDF for LaTeX
    if not args.fig_output.lower().endswith('.pdf'):
        args.fig_output = args.fig_output.rsplit('.', 1)[0] + '.pdf'
    
    # Get all files in the logs directory
    log_files = [f for f in os.listdir(args.logs_dir) if os.path.isfile(os.path.join(args.logs_dir, f))]
    log_files = sorted(log_files)
    
    if not log_files:
        print(f"No files found in {args.logs_dir}")
        return
    
    # Set up the plot with publication-ready parameters (using default fonts)
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    
    fig, ax = plt.subplots(figsize=args.figsize)
    colors = get_publication_colors(len(log_files))
    
    # First pass: collect all y-values to determine if log scale is needed
    all_y_values = []
    file_data = []
    
    for i, filename in enumerate(log_files):
        filepath = os.path.join(args.logs_dir, filename)
        
        # Parse the file
        data = parse_file(filepath, args.column)
        
        if len(data) == 0:
            print(f"Warning: No data found in {filename}")
            continue
        
        # Compute windowed statistics
        stats = compute_windowed_statistics(data, args.window_size, num_windows=args.num_windows)
        
        # Choose what to plot based on plot_type
        if args.plot_type == 'median':
            y_values = stats['medians']
            y_lower = stats['q25']
            y_upper = stats['q75']
            uncertainty_label = "IQR"
        elif args.plot_type == 'mean':
            y_values = stats['means']
            y_lower = stats['means'] - stats['stds']
            y_upper = stats['means'] + stats['stds']
            uncertainty_label = "±1σ"
        elif args.plot_type == 'p90':
            y_values = stats['p90']
            y_lower = stats['medians']
            y_upper = stats['p95']
            uncertainty_label = "P50-P95"
        elif args.plot_type == 'p95':
            y_values = stats['p95']
            y_lower = stats['p90']
            y_upper = stats['p95'] + (stats['p95'] - stats['p90'])
            uncertainty_label = "P90-P95+"
        elif args.plot_type == 'tail_avg':
            y_values = stats['tail_avgs']
            y_lower = stats['p90']
            y_upper = stats['tail_avgs'] + (stats['tail_avgs'] - stats['p90'])
            uncertainty_label = "P90-Top20%+"
        elif args.plot_type == 'std':
            y_values = stats['stds']
            y_lower = None
            y_upper = None
            uncertainty_label = ""
        
        all_y_values.append(y_values)
        file_data.append({
            'filename': filename,
            'y_values': y_values,
            'y_lower': y_lower,
            'y_upper': y_upper,
            'color': colors[i],
            'stats': stats
        })
    
    # Determine if log scale should be used
    use_log_scale = False
    if args.log_scale == 'on':
        use_log_scale = True
    elif args.log_scale == 'auto':
        use_log_scale = should_use_log_scale(all_y_values, threshold_factor=args.log_threshold)
    
    if use_log_scale:
        def log_formatter(x, p):
            if abs(x) < 1e3:  # Linear region - hide these labels
                return ''
            elif abs(x) >= 1e6:
                return f'{x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'{x/1e3:.0f}K'
            return format(int(x), ',')
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=10))
        ax.yaxis.set_minor_formatter(FuncFormatter(log_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
        ax.tick_params(axis='y', which='minor', labelsize=10)
        print(f"Using logarithmic y-axis scale (ratio threshold: {args.log_threshold})")
    
    # Second pass: plot the data
    for i, fd in enumerate(file_data):
        x = np.arange(len(fd['y_values']))
        
        # Format label with statistics
        stat_mean = np.mean(fd['y_values'])
        stat_std = np.std(fd['y_values'])
        label = f"{format_filename_for_label(fd['filename'])}"
        
        # Plot the main line
        ax.axhline(y=stat_mean, color=fd['color'], linestyle='--', linewidth=1.5, alpha=0.8)
        ax.plot(x, fd['y_values'], 'o-', color=fd['color'], label=label, 
                markersize=8, linewidth=2, markeredgewidth=1, markeredgecolor='white')
        
        # Plot uncertainty bands if requested (single horizontal band based on overall statistics)
        if args.show_uncertainty:
            if not use_log_scale:
                # Linear scale: use mean ± std
                ax.axhspan(stat_mean - stat_std, stat_mean + stat_std, color=fd['color'], alpha=0.15, zorder=1)
            else:
                # Log scale: use geometric mean and multiplicative std
                if np.all(fd['y_values'] > 0):
                    geom_mean = gmean(fd['y_values'])
                    geom_std = np.exp(np.std(np.log(fd['y_values'])))
                    ax.axhspan(geom_mean / geom_std, geom_mean * geom_std, color=fd['color'], alpha=0.15, zorder=1)
    
    # Customize the plot
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    
    # Set title
    if args.title:
        ax.set_title(args.title, pad=20)
    else:
        scale_text = " (log scale)" if use_log_scale else ""
        uncertainty_text = f" with {uncertainty_label}" if args.show_uncertainty else ""
        # ax.set_title(f'Windowed {args.plot_type.capitalize()} (window={args.window_size:,}){uncertainty_text}{scale_text}', 
        #             fontweight='bold', pad=20)
    
    # Improve legend
    legend = ax.legend(
        loc='center left', 
        bbox_to_anchor=(1.05, 0.5), 
        frameon=True, fancybox=True, shadow=True
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    
    # Improve grid and axes
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Use scientific notation for large numbers (but not in log scale)
    if not use_log_scale:
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    
    # Limit number of ticks to avoid crowding
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Save the plot with high quality settings for publication
    plt.savefig(args.fig_output, 
                dpi=300,
                bbox_inches='tight',
                facecolor='none',
                edgecolor='none',
                transparent=True,
                metadata={'Title': f'Windowed {args.plot_type.capitalize()} Analysis',
                         'Author': 'Generated by Enhanced Plotter',
                         'Subject': 'Statistical Analysis',
                         'Keywords': f'{args.plot_type}, windowed analysis, statistics'})
    
    print(f"High-quality plot saved to {args.fig_output}")
    print(f"Plot type: {args.plot_type}")
    print(f"Files processed: {len(log_files)}")
    print(f"Window size: {args.window_size:,}")
    if use_log_scale:
        min_val = np.min([np.min(y[y > 0]) for y in all_y_values if len(y[y > 0]) > 0])
        max_val = np.max([np.max(y) for y in all_y_values if len(y) > 0])
        print(f"Y-axis range: {min_val:.2e} to {max_val:.2e} (ratio: {max_val/min_val:.1f}x)")
    
    # Optionally show the plot
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
