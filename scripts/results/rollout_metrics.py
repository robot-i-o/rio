#!/usr/bin/env python3
"""
Script to summarize rollout results into a LaTeX table.
Reads JSON files from results_data/<robot_type>/<task_name>/*.json
and generates a LaTeX table with metrics.
"""

import json
from pathlib import Path

import numpy as np


def load_experiment_results(results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """
    Load all experiment results from the results_data directory.

    Args:
        results_dir: Path to results_data directory

    Returns:
        Dictionary mapping (robot_type, task_name) to list of trajectory stats
    """
    experiments = {}

    # Find all robot type directories
    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist")
        return experiments

    for robot_dir in results_dir.iterdir():
        if not robot_dir.is_dir():
            continue

        robot_type = robot_dir.name

        # Find all task directories
        for task_dir in robot_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name

            # Load all JSON files in this task directory
            trajectory_stats = []
            for json_file in task_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        trajectory_stats.append(data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

            # Include even if empty (to show '-' in table)
            experiments[(robot_type, task_name)] = trajectory_stats

    return experiments


def compute_metrics(trajectory_stats: list[dict]) -> dict[str, float]:
    """
    Compute aggregated metrics from a list of trajectory statistics.

    Args:
        trajectory_stats: List of trajectory stat dictionaries

    Returns:
        Dictionary with aggregated metrics
    """
    if not trajectory_stats:
        return {}

    # Extract metrics
    success_count = sum(1 for traj in trajectory_stats if traj.get("final_reward", 0) == 1.0)
    success_rate = (success_count / len(trajectory_stats)) * 100  # Percentage

    # Average task clock time
    times = [traj.get("total_time", 0) for traj in trajectory_stats if "total_time" in traj]
    avg_time = np.mean(times) if times else 0
    std_time = np.std(times) if times else 0

    # Average RAM memory (in GB)
    ram_mems = [traj.get("avg_ram_mem", 0) for traj in trajectory_stats if "avg_ram_mem" in traj]
    avg_ram = np.mean(ram_mems) if ram_mems else 0
    std_ram = np.std(ram_mems) if ram_mems else 0

    # Average CPU usage (percentage)
    cpu_usages = [traj.get("avg_cpu_usage", 0) for traj in trajectory_stats if "avg_cpu_usage" in traj]
    avg_cpu = np.mean(cpu_usages) if cpu_usages else 0
    std_cpu = np.std(cpu_usages) if cpu_usages else 0

    # Average GPU utilization (percentage)
    gpu_usages = [traj.get("avg_gpus_usage", 0) for traj in trajectory_stats if "avg_gpus_usage" in traj]
    avg_gpu_util = np.mean(gpu_usages) if gpu_usages else 0
    std_gpu_util = np.std(gpu_usages) if gpu_usages else 0

    # Average GPU memory utilization (percentage)
    gpu_mem_percs = [traj.get("avg_gpus_mem_percentage", 0) for traj in trajectory_stats if "avg_gpus_mem_percentage" in traj]
    avg_gpu_mem = np.mean(gpu_mem_percs) if gpu_mem_percs else 0
    std_gpu_mem = np.std(gpu_mem_percs) if gpu_mem_percs else 0

    return {
        "num_trajectories": len(trajectory_stats),
        "success_rate": success_rate,
        "avg_time": avg_time,
        "std_time": std_time,
        "avg_ram": avg_ram,
        "std_ram": std_ram,
        "avg_cpu": avg_cpu,
        "std_cpu": std_cpu,
        "avg_gpu_util": avg_gpu_util,
        "std_gpu_util": std_gpu_util,
        "avg_gpu_mem": avg_gpu_mem,
        "std_gpu_mem": std_gpu_mem,
    }


def generate_latex_table(experiments: dict[tuple[str, str], list[dict]]) -> str:
    """
    Generate a LaTeX table from experiment results.

    Args:
        experiments: Dictionary mapping (robot_type, task_name) to trajectory stats

    Returns:
        LaTeX table string
    """
    latex = []

    # Table header
    latex.append(r"\begin{table*}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{\textbf{Policy results.} \todo{PABLO / ARTHUR - Currently collecting}}")
    latex.append(r"\label{tab:policy_results}")
    latex.append(r"\begin{tabular}{l l c c c c c c}")
    latex.append(r"\hline")
    latex.append(r"Robot & Task & Success Rate & Time (s) & RAM (GB) & CPU (\%) & GPU Util (\%) & GPU Mem (\%) \\")
    latex.append(r"\hline")

    # Sort experiments by robot type and task name
    sorted_experiments = sorted(experiments.items(), key=lambda x: (x[0][0], x[0][1]))

    # Add rows
    for (robot_type, task_name), trajectory_stats in sorted_experiments:
        metrics = compute_metrics(trajectory_stats)

        # Format robot and task names (replace underscores with spaces or keep as is)
        robot_display = robot_type.replace("_", " ").title()
        task_display = task_name.replace("_", " ").title()

        # If no data, show '-' for all metrics
        if not metrics:
            row = f"{robot_display} & {task_display} & - & - & - & - & - & - \\\\"
            latex.append(row)
            continue

        # Format metrics with mean ± std where applicable
        success_rate = f"{metrics['success_rate']:.1f}"
        time_str = f"{metrics['avg_time']:.2f} $\\pm$ {metrics['std_time']:.2f}"
        ram_str = f"{metrics['avg_ram']:.1f} $\\pm$ {metrics['std_ram']:.1f}"
        cpu_str = f"{metrics['avg_cpu']:.1f} $\\pm$ {metrics['std_cpu']:.1f}"
        gpu_util_str = f"{metrics['avg_gpu_util']:.1f} $\\pm$ {metrics['std_gpu_util']:.1f}"
        gpu_mem_str = f"{metrics['avg_gpu_mem']:.1f} $\\pm$ {metrics['std_gpu_mem']:.1f}"

        row = (
            f"{robot_display} & {task_display} & {success_rate} & {time_str}"
            f" & {ram_str} & {cpu_str} & {gpu_util_str} & {gpu_mem_str} \\\\"
        )
        latex.append(row)

    # Table footer
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def main():
    """Main function to generate LaTeX table from rollout results."""
    # Get the project root directory (assuming script is in scripts/results/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    results_dir = project_root / "results_data"

    print(f"Loading results from: {results_dir}")

    # Load experiment results
    experiments = load_experiment_results(results_dir)

    if not experiments:
        print("No experiment results found!")
        return

    print(f"Found {len(experiments)} experiment(s):")
    for (robot_type, task_name), stats in experiments.items():
        count_msg = f"{len(stats)} trajectories" if stats else "empty (no data)"
        print(f"  - {robot_type}/{task_name}: {count_msg}")

    # Generate LaTeX table
    latex_table = generate_latex_table(experiments)

    # Print to stdout
    print("\n" + "=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    # Also save to file
    output_file = script_dir / "rollout_metrics_table.tex"
    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"\nTable saved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    for (robot_type, task_name), trajectory_stats in sorted(experiments.items()):
        metrics = compute_metrics(trajectory_stats)
        print(f"\n{robot_type}/{task_name}:")
        if not metrics:
            print("  No data available")
        else:
            print(f"  Trajectories: {metrics['num_trajectories']}")
            print(f"  Success Rate: {metrics['success_rate']:.1f}%")
            print(f"  Avg Time: {metrics['avg_time']:.2f}s (±{metrics['std_time']:.2f})")
            print(f"  Avg RAM: {metrics['avg_ram']:.1f}GB (±{metrics['std_ram']:.1f})")
            print(f"  Avg CPU: {metrics['avg_cpu']:.1f}% (±{metrics['std_cpu']:.1f})")
            print(f"  Avg GPU Util: {metrics['avg_gpu_util']:.1f}% (±{metrics['std_gpu_util']:.1f})")
            print(f"  Avg GPU Mem: {metrics['avg_gpu_mem']:.1f}% (±{metrics['std_gpu_mem']:.1f})")


if __name__ == "__main__":
    main()
