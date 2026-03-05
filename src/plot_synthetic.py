"""
Plot synthetic scaling experiment results.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(results_path: str):
    with open(results_path) as f:
        results = json.load(f)

    sizes = sorted(results.keys(), key=lambda k: results[k]["n_variables"])
    n_vars = [results[k]["n_variables"] for k in sizes]
    n_edges = [results[k]["n_edges"] for k in sizes]
    n_cands = [results[k]["n_candidates"] for k in sizes]
    n_mods = [results[k]["n_modifications"] for k in sizes]

    correct = [results[k]["correct_rate"] for k in sizes]
    mean_shd = [results[k]["mean_shd"] for k in sizes]
    std_shd = [results[k]["std_shd"] for k in sizes]
    mean_f1 = [results[k]["mean_f1"] for k in sizes]
    std_f1 = [results[k]["std_f1"] for k in sizes]
    mean_iters = [results[k]["mean_iterations"] for k in sizes]
    mean_wrmse = [results[k]["mean_weight_rmse"] for k in sizes]
    mean_wcov = [results[k]["mean_weight_coverage"] for k in sizes]
    conv_rate = [results[k]["convergence_rate"] for k in sizes]
    mean_map_prob = [results[k]["mean_map_prob"] for k in sizes]

    Path("plots").mkdir(exist_ok=True)

    # ---- Figure 1: Main 2x3 summary ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CBO Recovery on Synthetic Random DAGs (5–30 variables)",
                 fontsize=16, fontweight="bold")

    # (a) Correct recovery rate
    ax = axes[0, 0]
    colors = ["#2ecc71" if r >= 0.8 else "#e74c3c" if r < 0.5 else "#f39c12" for r in correct]
    ax.bar(range(len(sizes)), correct, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{v}" for v in n_vars])
    ax.set_xlabel("Variables")
    ax.set_ylabel("Correct Recovery Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("(a) Recovery Accuracy")
    for i, v in enumerate(correct):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")

    # (b) SHD
    ax = axes[0, 1]
    ax.errorbar(n_vars, mean_shd, yerr=std_shd, fmt="o-", color="#3498db",
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Variables")
    ax.set_ylabel("SHD")
    ax.set_title("(b) Structural Hamming Distance")

    # (c) F1
    ax = axes[0, 2]
    ax.errorbar(n_vars, mean_f1, yerr=std_f1, fmt="s-", color="#9b59b6",
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Variables")
    ax.set_ylabel("F1")
    ax.set_title("(c) Edge F1 Score")
    ax.set_ylim(0, 1.05)

    # (d) Iterations
    ax = axes[1, 0]
    ax.plot(n_vars, mean_iters, "D-", color="#1abc9c", linewidth=2, markersize=8)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Mean Iterations")
    ax.set_title("(d) Sample Efficiency")

    # (e) Weight RMSE
    ax = axes[1, 1]
    ax.plot(n_vars, mean_wrmse, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Weight RMSE")
    ax.set_title("(e) Weight Estimation Error")

    # (f) MAP probability & convergence
    ax = axes[1, 2]
    ax.plot(n_vars, mean_map_prob, "o-", color="#3498db", linewidth=2,
            markersize=8, label="P(MAP)")
    ax.plot(n_vars, conv_rate, "s--", color="#e67e22", linewidth=2,
            markersize=8, label="Conv. rate")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Probability / Rate")
    ax.set_title("(f) Posterior Concentration")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/synthetic_scaling.png", dpi=150, bbox_inches="tight")
    print("Saved plots/synthetic_scaling.png")

    # ---- Figure 2: Problem difficulty ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Scaling of Problem Difficulty", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(n_vars, n_edges, "o-", color="#3498db", linewidth=2, markersize=8)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Edges")
    ax.set_title("(a) Graph Density")

    ax = axes[1]
    ax.plot(n_vars, n_cands, "s-", color="#e74c3c", linewidth=2, markersize=8)
    ax.set_xlabel("Variables")
    ax.set_ylabel("K (candidates)")
    ax.set_title("(b) Candidate Set Size")

    ax = axes[2]
    ax.plot(n_vars, n_mods, "D-", color="#2ecc71", linewidth=2, markersize=8)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Edge Modifications")
    ax.set_title("(c) Candidate Modification Depth")

    plt.tight_layout()
    plt.savefig("plots/synthetic_difficulty.png", dpi=150, bbox_inches="tight")
    print("Saved plots/synthetic_difficulty.png")

    # ---- Figure 3: Posterior evolution per size ----
    n_panels = min(len(sizes), 7)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("Posterior Evolution by Graph Size", fontsize=14, fontweight="bold")

    for idx in range(n_panels):
        key = sizes[idx]
        ax = axes[idx]
        example = results[key].get("example_run", {})
        iters_data = example.get("iterations", [])
        if not iters_data:
            continue

        map_probs = [it["map_prob"] for it in iters_data]
        shds = [it["shd"] for it in iters_data]
        iter_nums = [it["iteration"] for it in iters_data]

        ax.plot(iter_nums, map_probs, "o-", color="#3498db", linewidth=2, markersize=5,
                label="P(MAP)")
        ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("P(MAP)")
        ax.set_title(f"n={results[key]['n_variables']}, e={results[key]['n_edges']}")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("plots/synthetic_posteriors.png", dpi=150, bbox_inches="tight")
    print("Saved plots/synthetic_posteriors.png")

    plt.close("all")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.plot_synthetic <results_json_path>")
        sys.exit(1)
    plot_results(sys.argv[1])
