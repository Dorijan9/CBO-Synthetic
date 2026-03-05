"""
CBO Synthetic Scaling Experiment.

Run CBO validation recovery on random DAGs from 5 to 30 variables.
Difficulty scales with size: more candidates, multi-edge modifications.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from src.scm import LinearGaussianSCM
from src.graph_belief import GraphBelief
from src.acquisition import select_intervention, entropy
from src.metrics import evaluate_graph, evaluate_weights
from src.synthetic_graphs import generate_scenario, SIZES


def run_cbo_single(size_key: str, config: dict, seed: int = 42,
                   verbose: bool = False) -> dict:
    """Run one CBO trial for a given graph size."""
    gt_path = f"data/{size_key}/ground_truth_dag.json"
    cand_path = f"data/{size_key}/candidate_graphs.json"

    # Load true weights
    with open(gt_path) as f:
        gt = json.load(f)
    true_weights = gt["scm_parameters"]["weights"]

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
    n_samples = config["n_int_samples"]
    intv_val = config["intervention_value"]

    obs_data = scm.sample_observational(config["n_obs_samples"], seed=seed)
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
        eval_s = evaluate_graph(gt_adj, map_adj)
        wp = belief.get_weight_posterior_summary(map_idx)
        eval_w = evaluate_weights(wp, true_weights)

        iterations.append({
            "iteration": t,
            "target": target,
            "map_graph": map_graph["id"],
            "map_prob": float(new_belief[map_idx]),
            "entropy": float(entropy(new_belief)),
            "shd": eval_s["shd"],
            "f1": eval_s["f1"],
            "weight_rmse": eval_w["weight_rmse"],
            "weight_coverage": eval_w["weight_coverage"],
        })

        if verbose:
            print(f"    Iter {t}: do({target}) MAP={map_graph['id']} "
                  f"P={new_belief[map_idx]:.3f} SHD={eval_s['shd']} F1={eval_s['f1']:.3f}")

        if belief.has_converged(threshold):
            if verbose:
                print(f"    *** Converged at iteration {t} ***")
            break

    final_idx = belief.map_estimate()
    final = belief.candidates[final_idx]
    final_eval = evaluate_graph(gt_adj, np.array(final["adjacency"]))
    final_wp = belief.get_weight_posterior_summary(final_idx)
    final_wt = evaluate_weights(final_wp, true_weights)
    correct = final["id"] == "G1"

    return {
        "size_key": size_key,
        "seed": seed,
        "n_variables": len(scm.variables),
        "n_candidates": belief.K,
        "iterations": iterations,
        "total_iterations": len(iterations),
        "converged": belief.has_converged(threshold),
        "correct_recovery": correct,
        "final_map_graph": final["id"],
        "final_map_prob": float(belief.belief[final_idx]),
        "final_shd": final_eval["shd"],
        "final_f1": final_eval["f1"],
        "final_precision": final_eval["precision"],
        "final_recall": final_eval["recall"],
        "final_weight_rmse": final_wt["weight_rmse"],
        "final_weight_coverage": final_wt["weight_coverage"],
        "entropy_reduction": float(entropy(belief.prior) - entropy(belief.belief)),
    }


def run_experiment(sizes: list = None, n_repeats: int = 5, seed: int = 42):
    """Run the full synthetic scaling experiment."""
    if sizes is None:
        sizes = SIZES

    # Scale config: more iterations for larger graphs, fewer EIG sims for speed
    base_config = {
        "sigma_w2": 0.5,
        "sigma_eps2": 0.3,
        "tau": 3.0,
        "n_obs_samples": 200,
        "n_int_samples": 10,
        "intervention_value": 2.0,
        "convergence_threshold": 0.90,
    }

    all_results = {}

    for n in sizes:
        size_key = f"{n}var"

        # Scale iterations and EIG budget with size
        config = {
            **base_config,
            "max_iterations": min(6 + n, 15),
            "n_eig_simulations": max(20, 100 - 3 * n),
            "n_obs_samples": min(150, max(30, 200 - 7 * n)),  # fewer obs for large n
            "n_int_samples": 10,
        }

        # Generate scenario (once per size)
        info = generate_scenario(n, seed=seed)

        print(f"\n{'='*60}")
        print(f"{size_key}: {info['n_variables']} vars, {info['n_edges']} edges, "
              f"{info['n_candidates']} candidates, mods={info['n_modifications']}")
        print(f"Max iterations: {config['max_iterations']}, "
              f"EIG sims: {config['n_eig_simulations']}")
        print(f"{'='*60}")

        runs = []
        for rep in range(n_repeats):
            rep_seed = seed + rep * 1000
            verbose = (rep == 0)
            if verbose:
                print(f"  Run {rep+1}/{n_repeats} (seed={rep_seed}):")

            result = run_cbo_single(size_key, config, seed=rep_seed, verbose=verbose)
            runs.append(result)

            if not verbose:
                status = "OK" if result["correct_recovery"] else "FAIL"
                print(f"  Run {rep+1}: {status} (SHD={result['final_shd']}, "
                      f"F1={result['final_f1']:.3f}, iters={result['total_iterations']})")

        # Aggregate
        agg = {
            "size_key": size_key,
            "n_variables": n,
            "n_edges": info["n_edges"],
            "n_candidates": info["n_candidates"],
            "n_modifications": info["n_modifications"],
            "max_iterations": config["max_iterations"],
            "n_repeats": n_repeats,
            "correct_rate": float(np.mean([r["correct_recovery"] for r in runs])),
            "mean_shd": float(np.mean([r["final_shd"] for r in runs])),
            "std_shd": float(np.std([r["final_shd"] for r in runs])),
            "mean_f1": float(np.mean([r["final_f1"] for r in runs])),
            "std_f1": float(np.std([r["final_f1"] for r in runs])),
            "mean_map_prob": float(np.mean([r["final_map_prob"] for r in runs])),
            "mean_iterations": float(np.mean([r["total_iterations"] for r in runs])),
            "mean_weight_rmse": float(np.mean([r["final_weight_rmse"] for r in runs])),
            "mean_weight_coverage": float(np.mean([r["final_weight_coverage"] for r in runs])),
            "convergence_rate": float(np.mean([r["converged"] for r in runs])),
            "example_run": runs[0],
        }
        all_results[size_key] = agg

        print(f"\n  SUMMARY: correct={agg['correct_rate']:.0%} "
              f"SHD={agg['mean_shd']:.2f}+/-{agg['std_shd']:.2f} "
              f"F1={agg['mean_f1']:.3f}+/-{agg['std_f1']:.3f} "
              f"iters={agg['mean_iterations']:.1f} "
              f"wRMSE={agg['mean_weight_rmse']:.4f} "
              f"wCov={agg['mean_weight_coverage']:.0%}")

    # Save
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/synthetic_results_{timestamp}.json"

    # Strip iteration details from non-example runs to save space
    save = {}
    for key, agg in all_results.items():
        save[key] = {k: v for k, v in agg.items()}

    with open(path, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nResults saved to {path}")

    # Final table
    print(f"\n{'='*90}")
    print(f"SYNTHETIC SCALING EXPERIMENT SUMMARY")
    print(f"{'='*90}")
    print(f"{'Size':<7} {'Vars':>4} {'Edges':>5} {'K':>3} {'Mods':>4} "
          f"{'Correct':>8} {'SHD':>10} {'F1':>10} {'Iters':>6} "
          f"{'wRMSE':>8} {'wCov':>6}")
    print("-" * 90)
    for n in sizes:
        key = f"{n}var"
        a = all_results[key]
        print(f"{key:<7} {a['n_variables']:>4} {a['n_edges']:>5} "
              f"{a['n_candidates']:>3} {a['n_modifications']:>4} "
              f"{a['correct_rate']:>7.0%} "
              f"{a['mean_shd']:>5.2f}+/-{a['std_shd']:.1f} "
              f"{a['mean_f1']:>5.3f}+/-{a['std_f1']:.2f} "
              f"{a['mean_iterations']:>5.1f} "
              f"{a['mean_weight_rmse']:>8.4f} "
              f"{a['mean_weight_coverage']:>5.0%}")

    return all_results, path


def main():
    results, path = run_experiment(sizes=[5, 7, 10, 15, 20, 25, 30], n_repeats=1, seed=42)


if __name__ == "__main__":
    main()
