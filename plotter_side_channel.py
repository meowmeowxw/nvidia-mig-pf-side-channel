import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

class SideChannelPlotter:
    def __init__(self, cache_dir="logs_side_channel_cache", baseline_file="logs_simple/inactive"):
        self.cache_dir = cache_dir
        self.baseline_file = baseline_file
        self.baseline_values = []
        self.cache_data = {}
        
        # Mapping from file ID to configuration label
        self.id_to_label = {
            3: 'ld.cv + st.wt (Host)',
            2: 'ld.cg + st.cg',
            4: 'ld.cv + st.cg',
            5: 'ld.cw + st.wb',
            6: 'ld.cg + atomicExch',
            7: 'nv_atomic_exch',
            8: 'atomicExchAdd',
            9: 'ld.ca + st.wb',
            10: 'ld.cg (constant) + st.cg'
        }
        
        # Custom ordering: put 3 before 2, then the rest in numerical order
        self.display_order = [3, 2, 4, 5, 6, 7, 8, 9, 10]
        
    def _apply_publication_style(self):
        """Apply publication-ready style to plots."""
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 12,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2,
        })
    
    def parse_baseline(self):
        """Parse the inactive baseline file."""
        try:
            with open(self.baseline_file, 'r') as f:
                lines = f.readlines()
                # Extract first column values (first 8000 points only)
                self.baseline_values = [float(line.split(',')[0].strip()) for line in lines if ',' in line][:8000]
            print(f"Loaded {len(self.baseline_values)} baseline values from {self.baseline_file}")
        except Exception as e:
            print(f"Error reading baseline file: {e}")
            
    def parse_cache_logs(self):
        """Parse all files in logs_side_channel_cache."""
        try:
            files = os.listdir(self.cache_dir)
            for filename in files:
                # Try to convert filename to integer
                try:
                    file_id = int(filename)
                except ValueError:
                    continue
                    
                filepath = os.path.join(self.cache_dir, filename)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    # Extract first column values (first 8000 points only)
                    values = [float(line.split(',')[0].strip()) for line in lines if ',' in line][:8000]
                    self.cache_data[file_id] = values
                    print(f"Loaded {len(values)} values from file {file_id}")
        except Exception as e:
            print(f"Error reading cache logs: {e}")
    
    def calculate_p99_overhead(self, values):
        """Calculate p99 overhead compared to baseline."""
        if not self.baseline_values or not values:
            return 0
        
        baseline_p99 = np.percentile(self.baseline_values, 99)
        values_p99 = np.percentile(values, 99)
        
        return values_p99 - baseline_p99
    
    def create_plot(self, output_file=None):
        """Create the bar plot showing p99 overhead."""
        self._apply_publication_style()
        
        # Use display_order instead of sorted IDs
        ordered_ids = [fid for fid in self.display_order if fid in self.cache_data]
        overheads = [self.calculate_p99_overhead(self.cache_data[file_id]) for file_id in ordered_ids]
        labels = [self.id_to_label.get(file_id, str(file_id)) for file_id in ordered_ids]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(max(12, len(ordered_ids) * 1.5), 7))
        
        x_positions = np.arange(len(ordered_ids))
        bars = ax.bar(x_positions, overheads, width=0.6, color='#a65628', 
                      alpha=0.85, edgecolor='black', linewidth=0.7)
        
        ax.set_ylabel('P99 Overhead (cycles)')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Set log scale on y-axis
        ax.set_yscale('log')
        
        # Add horizontal lines at powers of 10
        y_min, y_max = ax.get_ylim()
        powers_of_10 = [10**i for i in range(int(np.floor(np.log10(y_min))), int(np.ceil(np.log10(y_max))))]
        for power in powers_of_10:
            ax.axhline(y=power, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Disable automatic grid
        ax.grid(False)
        
        # Format y-axis with thousand separators
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.tight_layout()
        
        if output_file:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.savefig(output_file.replace('.png', '.pdf'), dpi=300, transparent=True, bbox_inches='tight')
            print(f"Saved plot to {output_file}")
            plt.close()
        else:
            plt.show()
    
    def run(self, output_file=None):
        """Main execution function."""
        print("Parsing baseline...")
        self.parse_baseline()
        
        print(f"\nParsing cache logs from {self.cache_dir}...")
        self.parse_cache_logs()
        
        print("\nCreating plot...")
        self.create_plot(output_file)
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Baseline P99: {np.percentile(self.baseline_values, 99):.0f} cycles")
        # Use display order for summary too
        for file_id in [fid for fid in self.display_order if fid in self.cache_data]:
            p99 = np.percentile(self.cache_data[file_id], 99)
            overhead = self.calculate_p99_overhead(self.cache_data[file_id])
            label = self.id_to_label.get(file_id, str(file_id))
            print(f"File {file_id} ({label}): P99 = {p99:.0f} cycles, Overhead = {overhead:.0f} cycles")

if __name__ == "__main__":
    plotter = SideChannelPlotter()
    plotter.run(output_file="figs/side_channel_p99_overhead.png")
