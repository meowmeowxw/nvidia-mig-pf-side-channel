#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator


COLUMN_WIDTH_IN = 3.35
PLOT_HEIGHT_IN = 2.75


class SideChannelPlotter:
    def __init__(self, cache_dir="logs_side_channel_cache", baseline_file="logs_simple/inactive"):
        self.script_dir = Path(__file__).resolve().parent
        self.cache_dir = self._resolve_input_path(cache_dir)
        self.baseline_file = self._resolve_input_path(baseline_file)
        self.baseline_values = []
        self.cache_data = {}

        # Mapping from file ID to configuration label.
        self.id_to_label = {
            3: "Host RAM (ld.cv + st.wt)",
            2: "ld.cg + st.cg",
            4: "ld.cv + st.cg",
            6: "ld.cg + atomicExch",
            7: "__nv_atomic_exchange",
            8: "atomicExch + atomicAdd",
            9: "ld.ca + st.wb",
            10: "ld.cg (constant) + st.cg",
        }

        self.display_order = [3, 2, 4, 6, 7, 8, 9, 10]

    def _resolve_input_path(self, path_str):
        candidate = Path(path_str)
        if candidate.exists():
            return candidate

        fallback = self.script_dir / path_str
        if fallback.exists():
            return fallback

        return candidate

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

    def parse_baseline(self):
        try:
            with self.baseline_file.open("r") as handle:
                lines = handle.readlines()
            self.baseline_values = [float(line.split(",")[0].strip()) for line in lines if "," in line][:8000]
            print(f"Loaded {len(self.baseline_values)} baseline values from {self.baseline_file}")
        except Exception as exc:
            print(f"Error reading baseline file: {exc}")

    def parse_cache_logs(self):
        try:
            for filepath in self.cache_dir.iterdir():
                filename = filepath.name
                try:
                    file_id = int(filename)
                except ValueError:
                    continue

                with filepath.open("r") as handle:
                    lines = handle.readlines()
                values = [float(line.split(",")[0].strip()) for line in lines if "," in line][:8000]
                self.cache_data[file_id] = values
                print(f"Loaded {len(values)} values from file {file_id}")
        except Exception as exc:
            print(f"Error reading cache logs: {exc}")

    def calculate_p99_overhead(self, values):
        if not self.baseline_values or not values:
            return 0

        baseline_p99 = np.percentile(self.baseline_values, 99)
        values_p99 = np.percentile(values, 99)
        return values_p99 - baseline_p99

    def _log_formatter(self, value, _):
        if value >= 1e6:
            return f"{value / 1e6:.1f}M"
        if value >= 1e3:
            value_in_k = value / 1e3
            if value >= 1e5 or abs(value_in_k - round(value_in_k)) < 1e-6:
                return f"{value_in_k:.0f}K"
            return f"{value_in_k:.1f}K"
        return f"{value:.0f}"

    def _rounded_axis_limit(self, value, step, round_fn):
        return step * round_fn(value / step)

    def _draw_axis_break_marks(self, left_ax, right_ax):
        marker_size = 0.013
        kwargs = dict(color="black", clip_on=False, linewidth=0.8)

        left_ax.plot((1 - marker_size, 1 + marker_size), (-marker_size, +marker_size),
                     transform=left_ax.transAxes, **kwargs)
        left_ax.plot((1 - marker_size, 1 + marker_size), (1 - marker_size, 1 + marker_size),
                     transform=left_ax.transAxes, **kwargs)
        right_ax.plot((-marker_size, +marker_size), (-marker_size, +marker_size),
                      transform=right_ax.transAxes, **kwargs)
        right_ax.plot((-marker_size, +marker_size), (1 - marker_size, 1 + marker_size),
                      transform=right_ax.transAxes, **kwargs)

    def _draw_segment(self, ax, y_positions, overheads, x_start, x_end, colors):
        visible_line_mask = overheads >= x_start
        if np.any(visible_line_mask):
            ax.hlines(
                y_positions[visible_line_mask],
                x_start,
                np.clip(overheads[visible_line_mask], x_start, x_end),
                color="#c7c7c7",
                linewidth=1.0,
                zorder=1,
            )

        visible_point_mask = (overheads >= x_start) & (overheads <= x_end)
        if np.any(visible_point_mask):
            ax.scatter(
                overheads[visible_point_mask],
                y_positions[visible_point_mask],
                s=42,
                color=np.array(colors)[visible_point_mask],
                edgecolors="black",
                linewidths=0.4,
                zorder=2,
            )

    def _configure_linear_axis(self, ax, x_min, x_max, num_ticks, show_y_ticks, ticks=None):
        ax.set_xlim(x_min, x_max)
        if ticks is None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
        else:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(self._log_formatter))
        ax.grid(True, axis="x", which="major", linestyle="--", alpha=0.35)
        ax.grid(False, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not show_y_ticks:
            ax.tick_params(axis="y", left=False, labelleft=False)

    def _create_log_plot(self, labels, overheads, y_positions, colors):
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN * 1.2, PLOT_HEIGHT_IN))

        positive_values = overheads[overheads > 0]
        min_positive = max(1.0, positive_values.min() * 0.85) if positive_values.size else 1.0

        ax.hlines(y_positions, min_positive, overheads, color="#c7c7c7", linewidth=1.0, zorder=1)
        ax.scatter(overheads, y_positions, s=42, color=colors, edgecolors="black", linewidths=0.4, zorder=2)

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        ax.xaxis.set_major_formatter(FuncFormatter(self._log_formatter))
        ax.set_xlabel("P99 overhead (cycles)")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.grid(True, axis="x", which="major", linestyle="--", alpha=0.35)
        ax.grid(False, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return fig

    def _create_broken_axis_plot(self, labels, overheads, y_positions, colors):
        fig, (ax_zoom, ax_outlier) = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(COLUMN_WIDTH_IN * 1.28, PLOT_HEIGHT_IN),
            gridspec_kw={"width_ratios": [4.4, 1.6], "wspace": 0.05},
        )

        outlier_value = overheads[0]
        clustered_values = overheads[1:]

        cluster_min = clustered_values.min()
        cluster_max = clustered_values.max()
        zoom_min = max(0.0, self._rounded_axis_limit(cluster_min * 0.9, 1000, np.floor))
        zoom_max = self._rounded_axis_limit(cluster_max * 1.12, 1000, np.ceil)
        outlier_min = self._rounded_axis_limit(outlier_value * 0.75, 50000, np.floor)
        outlier_max = self._rounded_axis_limit(outlier_value * 1.08, 50000, np.ceil)
        outlier_ticks = [self._rounded_axis_limit(outlier_value, 100000, np.round)]

        self._draw_segment(ax_zoom, y_positions, overheads, zoom_min, zoom_max, colors)
        self._draw_segment(ax_outlier, y_positions, overheads, outlier_min, outlier_max, colors)

        ax_zoom.set_yticks(y_positions)
        ax_zoom.set_yticklabels(labels)
        ax_zoom.invert_yaxis()
        ax_outlier.tick_params(axis="y", left=False, labelleft=False)

        self._configure_linear_axis(ax_zoom, zoom_min, zoom_max, num_ticks=4, show_y_ticks=True)
        self._configure_linear_axis(
            ax_outlier,
            outlier_min,
            outlier_max,
            num_ticks=3,
            show_y_ticks=False,
            ticks=outlier_ticks,
        )
        ax_zoom.spines["right"].set_visible(False)
        ax_outlier.spines["left"].set_visible(False)

        self._draw_axis_break_marks(ax_zoom, ax_outlier)
        fig.supxlabel("P99 overhead (cycles)")
        fig.subplots_adjust(left=0.44, right=0.98, top=0.98, bottom=0.17, wspace=0.05)

        return fig

    def create_plot(self, output_file):
        self._apply_publication_style()

        plot_rows = []
        for file_id in self.display_order:
            if file_id not in self.cache_data:
                continue
            plot_rows.append({
                "label": self.id_to_label.get(file_id, str(file_id)),
                "overhead": self.calculate_p99_overhead(self.cache_data[file_id]),
            })

        if not plot_rows:
            raise RuntimeError("No side-channel data found to plot.")

        plot_rows.sort(key=lambda row: row["overhead"], reverse=True)

        labels = [row["label"] for row in plot_rows]
        overheads = np.array([row["overhead"] for row in plot_rows], dtype=float)
        y_positions = np.arange(len(labels))

        base_color = "#4c78a8"
        colors = [base_color for _ in labels]

        outlier_ratio = overheads[0] / overheads[1] if len(overheads) > 1 and overheads[1] > 0 else np.inf
        if outlier_ratio >= 5:
            fig = self._create_broken_axis_plot(labels, overheads, y_positions, colors)
        else:
            fig = self._create_log_plot(labels, overheads, y_positions, colors)

        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            pdf_output = output_file if output_file.endswith(".pdf") else output_file.replace(".png", ".pdf")
            png_output = output_file if output_file.endswith(".png") else output_file.replace(".pdf", ".png")

            fig.patch.set_alpha(0.0)
            for axis in fig.axes:
                axis.patch.set_alpha(0.0)
            plt.savefig(pdf_output, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
            plt.savefig(png_output, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
            print(f"Saved plot to {pdf_output} and {png_output}")
            plt.close()
        else:
            plt.show()

    def run(self, output_file):
        print("Parsing baseline...")
        self.parse_baseline()

        print(f"\nParsing cache logs from {self.cache_dir}...")
        self.parse_cache_logs()

        print("\nCreating plot...")
        self.create_plot(output_file)

        print("\n=== Summary ===")
        print(f"Baseline P99: {np.percentile(self.baseline_values, 99):.0f} cycles")
        for file_id in self.display_order:
            if file_id not in self.cache_data:
                continue
            p99 = np.percentile(self.cache_data[file_id], 99)
            overhead = self.calculate_p99_overhead(self.cache_data[file_id])
            label = self.id_to_label.get(file_id, str(file_id))
            print(f"File {file_id} ({label}): P99 = {p99:.0f} cycles, Overhead = {overhead:.0f} cycles")


def parse_args():
    parser = argparse.ArgumentParser(description="Render the page-fault side-channel comparison plot.")
    parser.add_argument("--cache-dir", default="logs_side_channel_cache", help="Directory containing side-channel logs.")
    parser.add_argument("--baseline-file", default="logs_simple/inactive", help="Baseline log file.")
    parser.add_argument(
        "--output",
        default="figs/side_channel_p99_overhead.pdf",
        help="Output PDF or PNG path. The other format is written alongside it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    plotter = SideChannelPlotter(cache_dir=arguments.cache_dir, baseline_file=arguments.baseline_file)
    plotter.run(output_file=arguments.output)
