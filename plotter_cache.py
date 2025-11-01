import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import argparse
from matplotlib.ticker import FuncFormatter
import seaborn as sns # Added for the color palette function

class LogStatsVisualizer:
    def __init__(self, logs_dir="logs_cache", use_log_scale=False):
        self.logs_dir = logs_dir
        self.use_log_scale = use_log_scale
        self.data = defaultdict(lambda: defaultdict(list))
        self.baselines = {}  # Store inactive baselines for each config
        
        # Add mapping for legend labels
        self.config_label_mapping = {
            'l1': 'ld.ca + st.wb',
            'l2_ca_wb': 'ld.ca + st.wb',
            'l2_cg_cg': 'ld.cg + st.cg',
            'l2_cv_cg': 'ld.cv + st.cg',
            'l2_cg_atomicExch': 'ld.cg + atomicExch',
            'l2_atomicExchAdd': 'atomicExch + atomicAdd',
            'l2_atomicexch': '__nv_atomic_exchange',
            'l2_atomicexchange': 'nv_atomic_exchange',
            'l2_cache_constant_stcg': 'ld.cg (constant) + st.cg',
            'l2_cv_wt': 'ld.cv + st.wt (System RAM)',
            'uvm_load': 'UVM ld.cg',
        }

    def _apply_publication_style(self):
        """Apply a publication-ready style to plots, inspired by the reference script."""
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            # 'axes.titlesize': 18,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 12,
            # 'figure.titlesize': 20,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2,
        })

    def _get_publication_colors(self, n_colors):
        """Get a set of distinct, colorblind-friendly colors for publication (same as second script)."""
        if n_colors <= 8:
            # Using the exact color list provided by the user
            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
            return colors[:n_colors]
        else:
            # Use seaborn's husl palette for more colors, as in the reference
            return sns.color_palette("husl", n_colors)

    def get_display_label(self, config_name):
        """Get the display label for a configuration, using mapping if available."""
        # return self.config_label_mapping.get(config_name, config_name)
        return config_name

    def extract_workload_info(self, filename):
        """Extract workload type and parameters from filename."""
        if filename == 'inactive':
            return 'inactive', 0, 0
        elif filename == 'pytorch':
            return 'pytorch', 0, 0
        elif filename == 'copy_data':
            return 'copy_data', 0, 0
        elif filename.startswith('interrupt-storm-'):
            pattern = r'interrupt-storm-(\d+)x(\d+)'
            match = re.match(pattern, filename)
            if match:
                blocks = int(match.group(1))
                threads = int(match.group(2))
                return 'storm', blocks, threads
        return None, 0, 0

    def sort_workloads(self, workload_data):
        """Sort workloads according to the specified order."""
        def sort_key(item):
            # item can be just the workload_info tuple or a longer tuple
            workload_info = item[0] 
            workload, blocks, threads = workload_info
            if workload == 'inactive': return (0,)
            elif workload == 'pytorch': return (1,)
            elif workload == 'copy_data': return (2,)
            elif workload == 'storm':
                total_threads = blocks * threads
                return (3, total_threads, blocks)
            else: return (999,)
        return sorted(workload_data, key=sort_key)

    def create_workload_label(self, workload, blocks, threads):
        """Create a readable label for the workload."""
        if workload == 'inactive': return 'inactive'
        elif workload == 'pytorch': return 'pytorch'
        elif workload == 'copy_data': return 'copy_data'
        elif workload == 'storm': return f'PF-storm-{blocks}x{threads}'
        else: return 'unknown'

    def parse_logs(self):
        """Parse all log files and extract data."""
        log_files = glob.glob(os.path.join(self.logs_dir, "*"))
        for log_file in log_files:
            filename = os.path.basename(log_file)
            if not re.match(r'^\d+-.+', filename): continue
            
            workload = filename.split('-', 1)[1]
            try:
                with open(log_file, 'r') as f: lines = f.readlines()
                
                data_lines = [line.strip() for line in lines if ',' in line and any(c.isdigit() for c in line.split(',')[-1])]
                if len(data_lines) < 2: continue
                
                config_name = data_lines[0].split(',')[0].strip()
                if not config_name: continue

                values = [float(line.split(',')[-1].strip()) for line in data_lines[:] if line.split(',')[-1].strip().replace('.', '', 1).isdigit()]
                
                if values:
                    self.data[config_name][workload] = values
                    if workload == "inactive": self.baselines[config_name] = values
            except Exception as e:
                print(f"Error processing {log_file}: {e}")

    def calculate_statistic(self, values, stat_type):
        """Calculate the requested statistic."""
        if not values: return 0
        values = np.array(values)
        if stat_type == "median": return np.median(values)
        elif stat_type == "p90": return np.percentile(values, 90)
        elif stat_type == "p99": return np.percentile(values, 99)
        return np.mean(values)

    def calculate_error_bars(self, values):
        """Calculate Interquartile Range (IQR) for error bars."""
        if not values or len(values) < 2: return (0, 0)
        median, q1, q3 = np.median(values), np.percentile(values, 25), np.percentile(values, 75)
        return (median - q1, q3 - median)

    def calculate_overhead_cycles(self, values, baseline_values, stat_type):
        """Calculate overhead in absolute clock cycles."""
        if not baseline_values or not values: return 0
        current_stat = self.calculate_statistic(values, stat_type)
        baseline_stat = self.calculate_statistic(baseline_values, stat_type)
        return current_stat - baseline_stat
        
    def _create_grouped_plot(self, config_group, title, stat_type, output_dir):
        """Generates a single figure with grouped bars for a set of configurations."""
        
        # Filter data for the current group and sort the config names to ensure consistent color assignment
        sorted_config_group = sorted([cfg for cfg in config_group if cfg in self.data])
        group_data = {cfg: self.data[cfg] for cfg in sorted_config_group}
        
        if not group_data:
            print(f"Warning: No data found for any configuration in group '{title}'. Skipping plot.")
            return

        # Determine common, sorted workloads for the x-axis
        all_workload_infos = {}
        for config_name, workloads in group_data.items():
            for workload_name in workloads:
                if workload_name == "inactive": continue
                workload_info = self.extract_workload_info(workload_name)
                label = self.create_workload_label(*workload_info)
                if label not in all_workload_infos:
                    all_workload_infos[label] = workload_info
        
        sorted_workload_tuples = self.sort_workloads([(info,) for info in all_workload_infos.values()])
        sorted_labels = [self.create_workload_label(*info[0]) for info in sorted_workload_tuples]
        
        # Plotting Setup
        fig, ax = plt.subplots(figsize=(max(12, len(sorted_labels) * 1.8), 7))
        n_configs = len(group_data)
        n_workloads = len(sorted_labels)
        bar_width = 0.8 / n_configs
        x_positions = np.arange(n_workloads)
        colors = self._get_publication_colors(n_configs)

        # Plot bars for each configuration
        for i, config_name in enumerate(group_data.keys()):
            baseline_vals = self.baselines.get(config_name, [])
            overheads = []
            errors = []
            
            for label in sorted_labels:
                workload_found = False
                for workload_name, values in group_data[config_name].items():
                    if self.create_workload_label(*self.extract_workload_info(workload_name)) == label:
                        overhead = self.calculate_overhead_cycles(values, baseline_vals, stat_type)
                        overheads.append(overhead)
                        errors.append(self.calculate_error_bars(values) if stat_type == 'median' else (0, 0))
                        workload_found = True
                        break
                if not workload_found:
                    overheads.append(0)
                    errors.append((0, 0))
            
            pos = x_positions + (i - (n_configs - 1) / 2) * bar_width
            # Use the mapped label for the legend
            display_label = self.get_display_label(config_name)
            ax.bar(pos, overheads, bar_width, label=display_label, color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.7)
            
            if stat_type == 'median':
                y_errors = np.array(errors).T
                ax.errorbar(pos, overheads, yerr=y_errors, fmt='none', ecolor='black', capsize=4, elinewidth=1.5, alpha=0.7)
        
        # Apply log scale if enabled
        # Apply log scale if enabled
        if self.use_log_scale:
            # Use symlog with a linear threshold around zero
            ax.set_yscale('symlog', linthresh=1e3)
            ax.set_ylabel(f'{stat_type.capitalize()} Overhead (cycles, log scale)')
            
            # Custom formatter for cleaner labels
            def log_formatter(x, p):
                if abs(x) < 1e3:  # Linear region - hide these labels
                    return ''
                elif abs(x) >= 1e6:
                    return f'{x/1e6:.1f}M'
                elif abs(x) >= 1e3:
                    return f'{x/1e3:.0f}K'
                return format(int(x), ',')
            
            ax.get_yaxis().set_major_formatter(FuncFormatter(log_formatter))
            
            # Set custom tick locations to exclude ±100
            from matplotlib.ticker import FixedLocator
            # Get current ticks and filter out those in the linear region
            current_ylim = ax.get_ylim()
            # Create ticks for positive and negative sides
            pos_ticks = [1e3, 1e4, 1e5, 1e6, 1e7]
            neg_ticks = [-1e3, -1e4, -1e5, -1e6, -1e7]
            custom_ticks = [t for t in neg_ticks if t >= current_ylim[0]] + [0] + [t for t in pos_ticks if t <= current_ylim[1]]
            ax.yaxis.set_major_locator(FixedLocator(custom_ticks))
            
            # Adjust grid to show only major lines
            ax.grid(True, axis='y', which='major', linestyle='-', alpha=0.4, linewidth=0.8)
            ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.2, linewidth=0.5)
        else:
            ax.set_ylabel(f'{stat_type.capitalize()} Overhead (cycles)')
            ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Final Touches & Styling
        # ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sorted_labels, rotation=45, ha="right")
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.legend(title="Configuration")  # Changed title from "Workload" to "Configuration"
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', title)
            log_suffix = '_log' if self.use_log_scale else ''
            output_file = os.path.join(output_dir, f'{safe_title}_{stat_type}{log_suffix}.pdf')
            plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True, facecolor='none', edgecolor='none')
            print(f"Saved grouped plot: {output_file}")
            plt.close()
        else:
            plt.show()

    def create_grouped_comparison_plots(self, stat_type="mean", output_dir=None):
        """Creates grouped bar charts for predefined sets of configurations."""
        self._apply_publication_style()

        groups = {
            # "Memory Access Patterns": ['l2_ca_wb', 'l2_cg_cg', 'l2_cv_cg', 'l2_cg_atomicExch', 'l2_atomicexchange', 'l2_atomicExchAdd', 'l2_cache_constant_stcg'],
            # "PCIe Access Pattern": ['l2_cv_wt', 'uvm_load'],
            "gpu-internal-memory": ["ld.cg + st.cg",
                                    "ld.cv + st.cg",
                                    "ld.cv + st.wb",
                                    "ld.cg + atomicExch",
                                    "nv_atomic_exchange",
                                    "atomicExch + atomicAdd",
                                    "ld.ca + st.wb",
                                    "ld.cg (constant) + st.cg"],
            "gpu-host-memory": ["ld.cv + st.wt", "UVM ld.cg"],
            # "PCIe Access Pattern": ['uvm_load']
        }

        for title, config_group in groups.items():
            self._create_grouped_plot(config_group, title, stat_type, output_dir)
            
    def print_summary(self):
        """Print summary of parsed data."""
        print("\n=== Data Summary ===")
        for config_name in sorted(self.data.keys()):
            print(f'\nConfiguration "{config_name}":')
            for workload, values in self.data[config_name].items():
                print(f"  {workload}: {len(values)} data points")

    def run(self, stat_type="mean", output_dir=None):
        """Main execution function."""
        print(f"Parsing logs from {self.logs_dir}...")
        self.parse_logs()
        self.print_summary()
        scale_type = "logarithmic" if self.use_log_scale else "linear"
        print(f"\nCreating grouped comparison plots with {stat_type} statistics ({scale_type} scale)...")
        self.create_grouped_comparison_plots(stat_type, output_dir)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize log statistics with grouped overhead comparison')
    parser.add_argument('--statistics', '--stat', '-s', 
                       choices=['mean', 'median', 'p90', 'p99'],
                       default='median',
                       help='Statistic to calculate (default: median). Use "median" for error bars.')
    parser.add_argument('--output', '-o',
                       type=str,
                       default='figs',
                       help='Output directory for saving grouped plots (default: figs)')
    parser.add_argument('--logs-dir', '-d',
                       type=str,
                       default='logs_cache',
                       help='Directory containing log files (default: logs_cache)')
    parser.add_argument('--log_scale',
                       choices=['on', 'off'],
                       default='off',
                       help='Use logarithmic scale for y-axis (default: off)')
    return parser.parse_args()

def main():
    """Main function with command-line argument support."""
    try:
        args = parse_arguments()
        use_log_scale = (args.log_scale == 'on')
        visualizer = LogStatsVisualizer(args.logs_dir, use_log_scale)
        visualizer.run(args.statistics, args.output)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
