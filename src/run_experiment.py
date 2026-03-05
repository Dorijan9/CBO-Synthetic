"""
Synthetic CBO Scaling Experiment.

Sweep across graph sizes 5→30 with fixed edge density (~1.5 edges/var),
random weights, and K scaling with size. Measures how CBO recovery
degrades as the graph grows.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.scm import LinearGaussianSCM
from src.graph_belief import GraphBelief
from src.acquisition import select_intervention, entropy
from src.metrics import evaluate_graph, evaluate_weights
from src.synthetic_graphs import generate_scenario


# Graph sizes to test
SIZES = [5, 7, 10, 15, 20, 25, 30]


def run_cbo_single(size_label: str, true_weights: dict, config: dict,
                   seed: int = 42, verbose: bool = False) -> dict:
    """Run one CBO trial for a given graph size."""
    data_dir = "data"
    gt_path = f"{data_dir}/{size_label}/ground_truth_dag.json"
    cand_path = f"{data_dir}/{size_label}/candidate_graphs.json"

    scm = LinearGaussianSCM(dag_path=gt_path)
    belief = GraphBelief(
        candidates_path=cand_path,
        tau=config["tau"],
        sigma_w2=config["sigma_w2"],
        sigma_eps2=config["sigma_eps2"],
    )

    rng = np.random.default_rng(seed)
    max_iter = config["max_iterations"]
    threshold = config["convergence_threshold"]
    n_sim = config["n_eig_simulations"]
    n_samples = config["n_interventional_samples"]
    intv_val = config["intervention_value"]
    n_obs = config["n_observational_samples"]

    obs_data = scm.sample_observational(n_obs, seed=seed)
    all_intv = []
    iterations = []

    for t in range(1, max_iter + 1):
        iter_seed = rng.integers(0, 2**31)

        target, eig_scores = select_intervention(
            scm, belief, intv_val, n_sim, n_samples, seed=iter_seed
        )

        intv_data = scm.sample_interventional(
            target, intv_val, n_samples, seed=iter_seed + 1
        )
        all_intv.append(intv_data)
        combined = np.vstack([obs_data] + all_intv)

        old_belief = belief.belief.copy()
        new_belief = belief.update(intv_data, target, combined)

        map_idx = belief.map_estimate()
        map_graph = belief.candidates[map_idx]
        gt_adj = np.array(scm.adj)
        map_adj = np.array(map_graph["adjacency"])
        ev = evaluate_graph(gt_adj, map_adj)
        wp = belief.get_weight_posterior_summary(map_idx)
        wev = evaluate_weights(wp, true_weights)

        iterations.append({
            "iteration": t,
            "target": target,
            "map_graph": map_graph["id"],
            "map_prob": float(new_belief[map_idx]),
            "entropy": float(entropy(new_belief)),
            "shd": ev["shd"],
            "f1": ev["f1"],
            "weight_rmse": wev["weight_rmse"],
            "weight_coverage": wev["weight_coverage"],
        })

        if verbose:
            print(f"    Iter {t}: do({target}) → MAP={map_graph['id']} "
                  f"P={new_belief[map_idx]:.3f} SHD={ev['shd']} F1={ev['f1']:.3f}")

        if belief.has_converged(threshold):
            if verbose:
                print(f"    *** Converged at iteration {t} ***")
            break

    # Final
    final_idx = belief.map_estimate()
    final_map = belief.candidates[final_idx]
    final_ev = evaluate_graph(gt_adj, np.array(final_map["adjacency"]))
    final_wp = belief.get_weight_posterior_summary(final_idx)
    final_wev = evaluate_weights(final_wp, true_weights)

    return {
        "iterations": iterations,
        "total_iterations": len(iterations),
        "converged": belief.has_converged(threshold),
        "correct": final_map["id"] == "G1",
        "final_map": final_map["id"],
        "final_map_prob": float(belief.belief[final_idx]),
        "final_shd": final_ev["shd"],
        "final_f1": final_ev["f1"],
        "final_precision": final_ev["precision"],
        "final_recall": final_ev["recall"],
        "final_weight_rmse": final_wev["weight_rmse"],
        "final_weight_coverage": final_wev["weight_coverage"],
        "entropy_reduction": float(entropy(belief.prior) - entropy(belief.belief)),
    }


def run_synthetic_scaling(n_repeats: int = 5, seed: int = 42):
    """Run full synthetic scaling experiment."""
    config = {
        "sigma_eps2": 0.3,
        "sigma_w2": 0.5,
        "tau": 3.0,
        "n_observational_samples": 200,
        "n_interventional_samples": 10,
        "intervention_value": 2.0,
        "max_iterations": 10,
        "convergence_threshold": 0.90,
        "n_eig_simulations": 80,  # base; adapted per size below
    }

    all_results = {}

    print("=" * 80)
    print("SYNTHETIC CBO SCALING EXPERIMENT")
    print(f"Sizes: {SIZES}, repeats: {n_repeats}, edge density: 1.5")
    print("=" * 80)

    for n_vars in SIZES:
        # Adapt compute budget to graph size
        if n_vars <= 10:
            config["n_eig_simulations"] = 60
            config["max_iterations"] = 8
        elif n_vars <= 15:
            config["n_eig_simulations"] = 30
            config["max_iterations"] = 10
        elif n_vars <= 20:
            config["n_eig_simulations"] = 20
            config["max_iterations"] = 10
        else:
            config["n_eig_simulations"] = 15
            config["max_iterations"] = 12

        # Scale K with graph size
        K = min(6 + (n_vars - 5) // 3, 15)

        # Generate scenario (same DAG for all repeats of this size)
        info = generate_scenario(n_vars, edge_density=1.5, n_candidates=K, seed=seed)
        true_weights = info["true_weights"]

        size_results = []
        for rep in range(n_repeats):
            rep_seed = seed + rep * 1000 + n_vars * 100
            verbose = (rep == 0)
            if verbose:
                print(f"\n  n={n_vars} (K={K}, {info['n_edges']} edges), repeat {rep+1}/{n_repeats}:")
            result = run_cbo_single(
                info["size_label"], true_weights, config,
                seed=rep_seed, verbose=verbose
            )
            size_results.append(result)
            if not verbose:
                status = "✓" if result["correct"] else f"✗ (MAP={result['final_map']})"
                print(f"    repeat {rep+1}: {status} SHD={result['final_shd']} "
                      f"F1={result['final_f1']:.3f} iters={result['total_iterations']}")

        # Aggregate
        agg = {
            "n_vars": n_vars,
            "n_edges": info["n_edges"],
            "n_candidates": K,
            "edge_density": info["edge_density"],
            "n_repeats": n_repeats,
            "correct_rate": float(np.mean([r["correct"] for r in size_results])),
            "mean_shd": float(np.mean([r["final_shd"] for r in size_results])),
            "std_shd": float(np.std([r["final_shd"] for r in size_results])),
            "mean_f1": float(np.mean([r["final_f1"] for r in size_results])),
            "std_f1": float(np.std([r["final_f1"] for r in size_results])),
            "mean_map_prob": float(np.mean([r["final_map_prob"] for r in size_results])),
            "std_map_prob": float(np.std([r["final_map_prob"] for r in size_results])),
            "mean_iterations": float(np.mean([r["total_iterations"] for r in size_results])),
            "mean_weight_rmse": float(np.mean([r["final_weight_rmse"] for r in size_results])),
            "std_weight_rmse": float(np.std([r["final_weight_rmse"] for r in size_results])),
            "mean_weight_coverage": float(np.mean([r["final_weight_coverage"] for r in size_results])),
            "convergence_rate": float(np.mean([r["converged"] for r in size_results])),
            "mean_entropy_reduction": float(np.mean([r["entropy_reduction"] for r in size_results])),
            "example_run": size_results[0],
        }
        all_results[f"n{n_vars}"] = agg

        print(f"\n  → n={n_vars}: correct={agg['correct_rate']:.0%} "
              f"SHD={agg['mean_shd']:.2f}±{agg['std_shd']:.2f} "
              f"F1={agg['mean_f1']:.3f}±{agg['std_f1']:.3f} "
              f"iters={agg['mean_iterations']:.1f} "
              f"P(MAP)={agg['mean_map_prob']:.3f} "
              f"wRMSE={agg['mean_weight_rmse']:.4f} "
              f"wCov={agg['mean_weight_coverage']:.0%}")

    # Save
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"logs/synthetic_scaling_{timestamp}.json"

    # Strip individual runs for saving (keep example)
    save_data = {
        "config": config,
        "sizes": SIZES,
        "n_repeats": n_repeats,
        "seed": seed,
        "results": {},
    }
    for key, agg in all_results.items():
        save_copy = {k: v for k, v in agg.items()}
        # Keep only example run iterations, not all repeats
        save_data["results"][key] = save_copy

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Final summary table
    print(f"\n{'='*90}")
    print("SYNTHETIC SCALING SUMMARY")
    print(f"{'='*90}")
    print(f"{'N':>4} {'Edges':>5} {'K':>3} {'Correct':>8} {'SHD':>10} "
          f"{'F1':>10} {'P(MAP)':>10} {'Iters':>6} {'wRMSE':>10} {'wCov':>6} {'Conv':>6}")
    print("-" * 90)
    for n in SIZES:
        a = all_results[f"n{n}"]
        print(f"{n:>4} {a['n_edges']:>5} {a['n_candidates']:>3} "
              f"{a['correct_rate']:>7.0%} "
              f"{a['mean_shd']:>5.2f}±{a['std_shd']:<3.1f} "
              f"{a['mean_f1']:>5.3f}±{a['std_f1']:<4.2f} "
              f"{a['mean_map_prob']:>5.3f}±{a['std_map_prob']:<4.2f} "
              f"{a['mean_iterations']:>5.1f} "
              f"{a['mean_weight_rmse']:>5.4f}±{a['std_weight_rmse']:<4.2f} "
              f"{a['mean_weight_coverage']:>5.0%} "
              f"{a['convergence_rate']:>5.0%}")

    print(f"\nResults saved to {results_path}")
    return all_results, results_path


if __name__ == "__main__":
    run_synthetic_scaling(n_repeats=3, seed=42)
