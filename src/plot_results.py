"""
Plot synthetic scaling experiment results.

Usage: python -m src.plot_results <results_json_path>
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(results_path: str):
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    sizes = sorted(results.keys(), key=lambda k: results[k]["n_vars"])

    n_vars = [results[k]["n_vars"] for k in sizes]
    n_edges = [results[k]["n_edges"] for k in sizes]
    n_cands = [results[k]["n_candidates"] for k in sizes]

    correct = [results[k]["correct_rate"] for k in sizes]
    m_shd = [results[k]["mean_shd"] for k in sizes]
    s_shd = [results[k]["std_shd"] for k in sizes]
    m_f1 = [results[k]["mean_f1"] for k in sizes]
    s_f1 = [results[k]["std_f1"] for k in sizes]
    m_prob = [results[k]["mean_map_prob"] for k in sizes]
    s_prob = [results[k]["std_map_prob"] for k in sizes]
    m_iters = [results[k]["mean_iterations"] for k in sizes]
    m_wrmse = [results[k]["mean_weight_rmse"] for k in sizes]
    s_wrmse = [results[k]["std_weight_rmse"] for k in sizes]
    m_wcov = [results[k]["mean_weight_coverage"] for k in sizes]
    conv = [results[k]["convergence_rate"] for k in sizes]

    Path("plots").mkdir(exist_ok=True)

    # =========================================================================
    # Figure 1: Main 2x3 scaling dashboard
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CBO Recovery vs Graph Size (Synthetic Random DAGs, density≈1.5)",
                 fontsize=16, fontweight="bold")

    # (a) Correct recovery rate
    ax = axes[0, 0]
    colors = ["#2ecc71" if c >= 0.8 else "#f39c12" if c >= 0.5 else "#e74c3c" for c in correct]
    ax.bar(range(len(sizes)), correct, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{n}" for n in n_vars])
    ax.set_xlabel("Variables")
    ax.set_ylabel("Correct Recovery Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("(a) Recovery Accuracy")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    for i, v in enumerate(correct):
        ax.text(i, v + 0.03, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")

    # (b) SHD
    ax = axes[0, 1]
    ax.errorbar(n_vars, m_shd, yerr=s_shd, fmt="o-", color="#3498db",
                capsize=5, linewidth=2, markersize=7)
    ax.set_xlabel("Variables")
    ax.set_ylabel("SHD")
    ax.set_title("(b) Structural Hamming Distance")

    # (c) F1
    ax = axes[0, 2]
    ax.errorbar(n_vars, m_f1, yerr=s_f1, fmt="s-", color="#9b59b6",
                capsize=5, linewidth=2, markersize=7)
    ax.set_xlabel("Variables")
    ax.set_ylabel("F1 Score")
    ax.set_title("(c) Edge F1 Score")
    ax.set_ylim(0, 1.1)

    # (d) MAP probability
    ax = axes[1, 0]
    ax.errorbar(n_vars, m_prob, yerr=s_prob, fmt="D-", color="#e67e22",
                capsize=5, linewidth=2, markersize=7)
    ax.set_xlabel("Variables")
    ax.set_ylabel("P(MAP)")
    ax.set_title("(d) MAP Posterior Probability")
    ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.4, label="Threshold")
    ax.legend()

    # (e) Iterations
    ax = axes[1, 1]
    ax.plot(n_vars, m_iters, "o-", color="#1abc9c", linewidth=2, markersize=7)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Iterations")
    ax.set_title("(e) Mean Iterations to Termination")

    # (f) Weight RMSE
    ax = axes[1, 2]
    ax.errorbar(n_vars, m_wrmse, yerr=s_wrmse, fmt="^-", color="#e74c3c",
                capsize=5, linewidth=2, markersize=7)
    ax.set_xlabel("Variables")
    ax.set_ylabel("Weight RMSE")
    ax.set_title("(f) Weight Estimation Error")

    plt.tight_layout()
    plt.savefig("plots/synthetic_scaling_dashboard.png", dpi=150, bbox_inches="tight")
    print("Saved plots/synthetic_scaling_dashboard.png")

    # =========================================================================
    # Figure 2: Complexity annotation (edges + K alongside performance)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Graph Complexity vs Recovery", fontsize=14, fontweight="bold")

    # Edges vs SHD
    ax = axes[0]
    ax.scatter(n_edges, m_shd, s=100, c=n_vars, cmap="viridis", edgecolors="black", zorder=5)
    for i, n in enumerate(n_vars):
        ax.annotate(f"n={n}", (n_edges[i], m_shd[i]), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Mean SHD")
    ax.set_title("(a) Edges vs Structural Error")

    # K vs correct rate
    ax = axes[1]
    ax.scatter(n_cands, correct, s=100, c=n_vars, cmap="viridis", edgecolors="black", zorder=5)
    for i, n in enumerate(n_vars):
        ax.annotate(f"n={n}", (n_cands[i], correct[i]), textcoords="offset points",
                    xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Number of Candidates (K)")
    ax.set_ylabel("Correct Recovery Rate")
    ax.set_title("(b) Candidate Pool Size vs Accuracy")
    ax.set_ylim(0, 1.1)

    # Coverage vs size
    ax = axes[2]
    ax.plot(n_vars, m_wcov, "s-", color="#2ecc71", linewidth=2, markersize=8)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="Nominal 95%")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Weight 95% CI Coverage")
    ax.set_title("(c) Credible Interval Calibration")
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/synthetic_complexity.png", dpi=150, bbox_inches="tight")
    print("Saved plots/synthetic_complexity.png")

    # =========================================================================
    # Figure 3: Posterior evolution per size (example runs)
    # =========================================================================
    n_plots = len(sizes)
    fig, axes = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Posterior Evolution by Graph Size (Example Runs)", fontsize=14, fontweight="bold")

    for idx, key in enumerate(sizes):
        ax = axes[idx]
        example = results[key].get("example_run", {})
        iters_data = example.get("iterations", [])
        if not iters_data:
            continue

        iter_nums = [it["iteration"] for it in iters_data]
        map_probs = [it["map_prob"] for it in iters_data]
        shds = [it["shd"] for it in iters_data]

        ax.plot(iter_nums, map_probs, "o-", linewidth=2, markersize=5, color="#3498db", label="P(MAP)")
        ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("P(MAP)")
        n = results[key]["n_vars"]
        e = results[key]["n_edges"]
        ax.set_title(f"n={n}, e={e}", fontsize=11)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("plots/synthetic_posteriors.png", dpi=150, bbox_inches="tight")
    print("Saved plots/synthetic_posteriors.png")

    plt.close("all")
    print("\nAll plots saved to plots/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.plot_results <results_json_path>")
        sys.exit(1)
    plot_results(sys.argv[1])
