#!/usr/bin/env python3

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import FixedLocator, FuncFormatter


COLUMN_WIDTH_IN = 3.35
FULL_WIDTH_IN = 8.4
HOST_HEIGHT_IN = 3.35
HEATMAP_HEIGHT_IN = 1.8
SINGLE_COLUMN_HEATMAP_HEIGHT_IN = 2.3


class LogStatsVisualizer:
    def __init__(self, logs_dir="logs_cache", use_log_scale=False):
        self.logs_dir = logs_dir
        self.use_log_scale = use_log_scale
        self.data = defaultdict(lambda: defaultdict(list))
        self.baselines = {}

        self.workload_label_mapping = {
            "inactive": "Inactive",
            "pytorch": "PyTorch",
            "copy_data": "copy_data",
        }
        self.config_axis_labels = {
            "ld.ca + st.wb": "ld.ca + st.wb",
            "ld.cg + st.cg": "ld.cg + st.cg",
            "ld.cv + st.cg": "ld.cv + st.cg",
            "ld.cg + atomicExch": "ld.cg + atomicExch",
            "nv_atomic_exchange": "__nv_atomic_exchange",
            "atomicExch + atomicAdd": "atomicExch + atomicAdd",
            "ld.cg (constant) + st.cg": "ld.cg (constant) + st.cg",
            "ld.cv + st.wt": "ld.cv + st.wt",
            "UVM ld.cg": "UVM ld.cg",
        }
        self.group_specs = {
            "gpu-host-memory": {
                "stem": "gpu-host-memory",
                "configs": ["ld.cv + st.wt", "UVM ld.cg"],
                "kind": "dumbbell",
            },
            "gpu-internal-memory": {
                "stem": "Memory_Access_Patterns",
                "configs": [
                    "ld.ca + st.wb",
                    "ld.cg + st.cg",
                    "ld.cv + st.cg",
                    "ld.cg + atomicExch",
                    "nv_atomic_exchange",
                    "atomicExch + atomicAdd",
                    "ld.cg (constant) + st.cg",
                ],
                "kind": "heatmap",
            },
        }

    def _apply_publication_style(self):
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    def extract_workload_info(self, filename):
        if filename == "inactive":
            return ("inactive", 0, 0)
        if filename == "pytorch":
            return ("pytorch", 0, 0)
        if filename == "copy_data":
            return ("copy_data", 0, 0)
        if filename.startswith("interrupt-storm-"):
            match = re.match(r"interrupt-storm-(\d+)x(\d+)", filename)
            if match:
                return ("storm", int(match.group(1)), int(match.group(2)))
        return (None, 0, 0)

    def sort_workloads(self, workload_infos):
        def sort_key(info):
            workload, blocks, threads = info
            if workload == "inactive":
                return (0,)
            if workload == "pytorch":
                return (1,)
            if workload == "copy_data":
                return (2,)
            if workload == "storm":
                return (3, blocks * threads, blocks)
            return (99,)

        return sorted(workload_infos, key=sort_key)

    def create_workload_label(self, workload, blocks, threads, multiline=False):
        if workload == "storm":
            return f"PF\n{blocks}x{threads}" if multiline else f"PF{blocks}x{threads}"
        return self.workload_label_mapping.get(workload, workload)

    def parse_logs(self):
        for log_file in glob.glob(os.path.join(self.logs_dir, "*")):
            filename = os.path.basename(log_file)
            if not re.match(r"^\d+-.+", filename):
                continue

            workload_name = filename.split("-", 1)[1]
            try:
                with open(log_file, "r") as handle:
                    lines = handle.readlines()

                data_lines = [
                    line.strip()
                    for line in lines
                    if "," in line and any(char.isdigit() for char in line.split(",")[-1])
                ]
                if len(data_lines) < 2:
                    continue

                config_name = data_lines[0].split(",")[0].strip()
                if not config_name:
                    continue

                values = []
                for line in data_lines:
                    sample = line.split(",")[-1].strip()
                    try:
                        values.append(float(sample))
                    except ValueError:
                        continue

                if not values:
                    continue

                self.data[config_name][workload_name] = values
                if workload_name == "inactive":
                    self.baselines[config_name] = values
            except Exception as exc:
                print(f"Error processing {log_file}: {exc}")

    def calculate_statistic(self, values, stat_type):
        if not values:
            return 0.0
        values = np.array(values)
        if stat_type == "median":
            return float(np.median(values))
        if stat_type == "p90":
            return float(np.percentile(values, 90))
        if stat_type == "p99":
            return float(np.percentile(values, 99))
        return float(np.mean(values))

    def calculate_overhead_cycles(self, values, baseline_values, stat_type):
        if not baseline_values or not values:
            return 0.0
        current_stat = self.calculate_statistic(values, stat_type)
        baseline_stat = self.calculate_statistic(baseline_values, stat_type)
        return current_stat - baseline_stat

    def _collect_workloads(self, group_data):
        workload_infos = set()
        for workloads in group_data.values():
            for workload_name in workloads:
                if workload_name == "inactive":
                    continue
                workload_info = self.extract_workload_info(workload_name)
                if workload_info[0] is not None:
                    workload_infos.add(workload_info)
        return self.sort_workloads(list(workload_infos))

    def _build_overhead_lookup(self, config_name, stat_type):
        lookup = {}
        baseline_values = self.baselines.get(config_name, [])
        for workload_name, values in self.data[config_name].items():
            workload_info = self.extract_workload_info(workload_name)
            if workload_info[0] is None or workload_name == "inactive":
                continue
            lookup[workload_info] = self.calculate_overhead_cycles(values, baseline_values, stat_type)
        return lookup

    def _save_figure(self, fig, output_dir, stem, stat_type=None, use_log_suffix=False):
        os.makedirs(output_dir, exist_ok=True)
        log_suffix = "_log" if use_log_suffix else ""
        filename_stem = f"{stem}_{stat_type}" if stat_type else stem
        pdf_output = os.path.join(output_dir, f"{filename_stem}{log_suffix}.pdf")
        png_output = os.path.join(output_dir, f"{filename_stem}{log_suffix}.png")
        fig.patch.set_alpha(0.0)
        for axis in fig.axes:
            axis.patch.set_alpha(0.0)
        fig.savefig(pdf_output, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
        fig.savefig(png_output, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
        print(f"Saved {pdf_output} and {png_output}")
        plt.close(fig)

    def _symlog_formatter(self, value, _):
        if value == 0:
            return "0"
        sign = "-" if value < 0 else ""
        value = abs(value)
        if value >= 1e6:
            return f"{sign}{value / 1e6:.1f}M"
        if value >= 1e3:
            return f"{sign}{value / 1e3:.0f}K"
        return f"{sign}{value:.0f}"

    def _style_host_memory_axis(self, ax, stat_type, y_positions, labels, flat_values):
        if self.use_log_scale:
            ax.set_xscale("symlog", linthresh=1e3)
            tick_candidates = np.array([-1e4, -1e3, 0, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=float)
            xmin = flat_values.min() * 1.15 if flat_values.min() < 0 else -1.2e3
            xmax = flat_values.max() * 1.2 if flat_values.max() > 0 else 1.2e3
            visible_ticks = [tick for tick in tick_candidates if xmin <= tick <= xmax]
            if 0 not in visible_ticks:
                visible_ticks.append(0.0)
                visible_ticks.sort()
            ax.xaxis.set_major_locator(FixedLocator(visible_ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(self._symlog_formatter))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

        ax.axvline(0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xlabel(f"{stat_type.upper()} overhead (cycles)")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        ax.grid(False, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _create_host_memory_dumbbell_plot(self, ordered_configs, ordered_workloads, labels, lookups, stat_type, output_dir, stem):
        y_positions = np.arange(len(labels))
        series = [[lookups[cfg].get(info, 0.0) for info in ordered_workloads] for cfg in ordered_configs]
        flat_values = np.array(series).flatten()

        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN * 1.18, HOST_HEIGHT_IN))
        colors = ["#4c78a8", "#f58518"]
        markers = ["o", "s"]

        for idx, _ in enumerate(ordered_workloads):
            ax.plot([series[0][idx], series[1][idx]], [y_positions[idx], y_positions[idx]], color="#c7c7c7", linewidth=1.0, zorder=1)

        for cfg_index, config_name in enumerate(ordered_configs):
            ax.scatter(
                series[cfg_index],
                y_positions,
                s=38,
                marker=markers[cfg_index],
                color=colors[cfg_index],
                edgecolors="black",
                linewidths=0.4,
                label=self.config_axis_labels.get(config_name, config_name),
                zorder=2,
            )

        self._style_host_memory_axis(ax, stat_type, y_positions, labels, flat_values)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save_figure(fig, output_dir, stem, stat_type, use_log_suffix=self.use_log_scale)

    def _create_host_memory_offset_plot(self, ordered_configs, ordered_workloads, labels, lookups, stat_type, output_dir, stem):
        y_positions = np.arange(len(labels))
        series = [[lookups[cfg].get(info, 0.0) for info in ordered_workloads] for cfg in ordered_configs]
        flat_values = np.array(series).flatten()

        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN * 1.18, HOST_HEIGHT_IN))
        colors = ["#4c78a8", "#f58518"]
        markers = ["o", "s"]
        y_offsets = [-0.16, 0.16]

        for idx, y in enumerate(y_positions):
            if idx % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f5f5f5", zorder=0)

        for cfg_index, config_name in enumerate(ordered_configs):
            ax.scatter(
                series[cfg_index],
                y_positions + y_offsets[cfg_index],
                s=42,
                marker=markers[cfg_index],
                color=colors[cfg_index],
                edgecolors="black",
                linewidths=0.4,
                label=self.config_axis_labels.get(config_name, config_name),
                zorder=2.5,
            )

        self._style_host_memory_axis(ax, stat_type, y_positions, labels, flat_values)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        self._save_figure(fig, output_dir, stem, stat_type, use_log_suffix=self.use_log_scale)

    def _create_host_memory_plot(self, config_group, stat_type, output_dir, stem):
        ordered_configs = [cfg for cfg in config_group if cfg in self.data]
        if len(ordered_configs) < 2:
            print(f"Warning: not enough host-memory configurations found for {stem}.")
            return

        group_data = {cfg: self.data[cfg] for cfg in ordered_configs}
        ordered_workloads = self._collect_workloads(group_data)
        lookups = {cfg: self._build_overhead_lookup(cfg, stat_type) for cfg in ordered_configs}
        labels = [self.create_workload_label(*info) for info in ordered_workloads]

        self._create_host_memory_dumbbell_plot(
            ordered_configs,
            ordered_workloads,
            labels,
            lookups,
            stat_type,
            output_dir,
            stem,
        )
        self._create_host_memory_offset_plot(
            ordered_configs,
            ordered_workloads,
            labels,
            lookups,
            stat_type,
            output_dir,
            f"{stem}_offset",
        )

    def _create_internal_memory_heatmap_variant(self, config_group, stat_type, output_dir, stem, height_in):
        ordered_configs = [cfg for cfg in config_group if cfg in self.data]
        group_data = {cfg: self.data[cfg] for cfg in ordered_configs}
        if not group_data:
            print(f"Warning: no internal-memory data found for {stem}.")
            return

        ordered_workloads = self._collect_workloads(group_data)
        workload_labels = [self.create_workload_label(*info, multiline=False) for info in ordered_workloads]
        kernel_labels = [self.config_axis_labels.get(cfg, cfg) for cfg in ordered_configs]
        lookups = {cfg: self._build_overhead_lookup(cfg, stat_type) for cfg in ordered_configs}
        matrix = np.array([[lookups[cfg].get(info, 0.0) for info in ordered_workloads] for cfg in ordered_configs], dtype=float)

        max_positive = max(float(np.max(matrix)), 1.0)
        # Negative overheads are small and visually distracting in this figure.
        # Clamp them to the baseline color and spend the palette on positive overheads.
        norm = colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=max_positive)
        cmap = colors.LinearSegmentedColormap.from_list(
            "paper_reds",
            [
                (0.0, "#fff5f0"),
                (0.3, "#fcbba1"),
                (0.55, "#fb6a4a"),
                (0.78, "#de2d26"),
                (1.0, "#b30000"),
            ],
        )
        cmap.set_under("#f2f2f2")

        fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, height_in))
        heatmap = sns.heatmap(
            matrix,
            cmap=cmap,
            norm=norm,
            linewidths=0.35,
            linecolor="white",
            cbar_kws={
                "label": f"{stat_type.upper()} overhead (cycles)",
                "shrink": 0.9,
                "pad": 0.008,
                "fraction": 0.03,
            },
            xticklabels=workload_labels,
            yticklabels=kernel_labels,
            ax=ax,
        )

        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, labelsize=7.1, pad=1.4, length=0)
        ax.tick_params(axis="y", rotation=0, labelsize=7.2, pad=0.5, length=0)
        colorbar = heatmap.collections[0].colorbar
        if max_positive <= 20:
            tick_candidates = [0, 5, 10, 15, 20]
        elif max_positive <= 100:
            tick_candidates = [0, 10, 25, 50, 75, 100]
        else:
            tick_candidates = [0, 50, 100, 250, 500]
        visible_ticks = [tick for tick in tick_candidates if tick <= max_positive]
        if not visible_ticks or visible_ticks[-1] != int(max_positive):
            visible_ticks.append(int(round(max_positive)))
        colorbar.set_ticks(sorted(set(visible_ticks)))
        colorbar.set_ticklabels([f"{tick:d}" for tick in sorted(set(visible_ticks))])
        colorbar.ax.tick_params(labelsize=7, length=0)
        plt.tight_layout(pad=0.05)
        self._save_figure(fig, output_dir, stem, stat_type=stat_type, use_log_suffix=False)

    def _create_internal_memory_heatmap(self, config_group, stat_type, output_dir, stem):
        self._create_internal_memory_heatmap_variant(
            config_group,
            stat_type,
            output_dir,
            stem,
            HEATMAP_HEIGHT_IN,
        )
        self._create_internal_memory_heatmap_variant(
            config_group,
            stat_type,
            output_dir,
            f"{stem}_singlecol",
            SINGLE_COLUMN_HEATMAP_HEIGHT_IN,
        )

    def create_grouped_comparison_plots(self, stat_type="mean", output_dir=None):
        self._apply_publication_style()

        for group_name, spec in self.group_specs.items():
            configs = spec["configs"]
            if spec["kind"] == "dumbbell":
                self._create_host_memory_plot(configs, stat_type, output_dir, spec["stem"])
            else:
                self._create_internal_memory_heatmap(configs, stat_type, output_dir, spec["stem"])

    def print_summary(self):
        print("\n=== Data Summary ===")
        for config_name in sorted(self.data.keys()):
            print(f'\nConfiguration "{config_name}":')
            for workload, values in self.data[config_name].items():
                print(f"  {workload}: {len(values)} data points")

    def run(self, stat_type="mean", output_dir=None):
        print(f"Parsing logs from {self.logs_dir}...")
        self.parse_logs()
        self.print_summary()
        print(f"\nCreating publication plots with {stat_type} statistics...")
        self.create_grouped_comparison_plots(stat_type, output_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize cache-side interference with publication-oriented layouts.")
    parser.add_argument(
        "--statistics",
        "--stat",
        "-s",
        choices=["mean", "median", "p90", "p99"],
        default="median",
        help='Statistic to calculate (default: "median").',
    )
    parser.add_argument("--output", "-o", type=str, default="figs", help="Output directory for figures.")
    parser.add_argument("--logs-dir", "-d", type=str, default="logs_cache", help="Directory containing log files.")
    parser.add_argument(
        "--log_scale",
        choices=["on", "off"],
        default="off",
        help="Use symlog on the host-memory comparison axis.",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        visualizer = LogStatsVisualizer(args.logs_dir, use_log_scale=(args.log_scale == "on"))
        visualizer.run(args.statistics, args.output)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as exc:
        print(f"An error occurred: {exc}")


if __name__ == "__main__":
    main()
