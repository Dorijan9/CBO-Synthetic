"""
Synthetic random DAG generator for scaling experiments.

Generates random DAGs with:
- Controlled number of variables (n)
- Controlled edge density (edges_per_var, default 1.5)
- Random weights drawn from [-0.8, -0.3] ∪ [+0.3, +0.8]
- Candidate graphs with multi-edge modifications (harder than single-edge)

Candidate generation scales K with graph size to maintain difficulty.
"""

import json
import numpy as np
from pathlib import Path


def random_dag(n: int, edges_per_var: float = 1.5, seed: int = 42) -> tuple:
    """Generate a random DAG with n variables and ~n*edges_per_var edges.

    Strategy: fix a topological order (0, 1, ..., n-1), then randomly
    add forward edges (i->j where i < j) until target count reached.

    Returns:
        variables: list of variable names ["X0", "X1", ...]
        topo_order: topological ordering
        adj: (n, n) adjacency matrix
        edges: list of (source, target) tuples
    """
    rng = np.random.default_rng(seed)
    variables = [f"X{i}" for i in range(n)]
    topo_order = list(variables)

    target_edges = int(round(n * edges_per_var))

    # All possible forward edges
    possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(possible)

    target_edges = min(target_edges, len(possible))
    selected = possible[:target_edges]

    adj = np.zeros((n, n), dtype=int)
    edges = []
    for i, j in selected:
        adj[i, j] = 1
        edges.append((variables[i], variables[j]))

    return variables, topo_order, adj, edges


def assign_weights(edges: list, seed: int = 42) -> dict:
    """Assign random weights to edges.

    Weights drawn from [-0.8, -0.3] U [+0.3, +0.8] to ensure
    non-trivial effects with clear sign.
    """
    rng = np.random.default_rng(seed)
    weights = {}
    for src, tgt in edges:
        if rng.random() < 0.3:
            w = rng.uniform(-0.8, -0.3)
        else:
            w = rng.uniform(0.3, 0.8)
        weights[(src, tgt)] = round(w, 3)
    return weights


def _has_cycle(adj: np.ndarray) -> bool:
    """Check for cycles via DFS."""
    n = adj.shape[0]
    WHITE, GREY, BLACK = 0, 1, 2
    color = [WHITE] * n

    def dfs(u):
        color[u] = GREY
        for v in range(n):
            if adj[u, v]:
                if color[v] == GREY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in range(n))


def generate_candidates(variables: list, adj_true: np.ndarray, edges: list,
                        n_candidates: int = 8, n_modifications: int = 2,
                        seed: int = 42) -> list:
    """Generate candidate graphs with multi-edge modifications.

    G1 = ground truth
    G2..GK = each differs from truth by n_modifications edge changes
    """
    rng = np.random.default_rng(seed)
    n = len(variables)

    candidates = [{
        "id": "G1",
        "confidence": 0.70,
        "description": "Ground truth",
        "adjacency": adj_true.tolist(),
    }]

    confidences = np.linspace(0.60, 0.35, n_candidates - 1)
    attempts = 0
    max_attempts = 1000

    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        adj_mod = adj_true.copy()

        for _ in range(n_modifications):
            op = rng.choice(["remove", "reverse", "add"], p=[0.35, 0.30, 0.35])

            if op == "remove":
                existing = list(zip(*np.where(adj_mod != 0)))
                if existing:
                    idx = rng.integers(len(existing))
                    i, j = existing[idx]
                    adj_mod[i, j] = 0

            elif op == "reverse":
                existing = list(zip(*np.where(adj_mod != 0)))
                if existing:
                    idx = rng.integers(len(existing))
                    i, j = existing[idx]
                    adj_mod[i, j] = 0
                    adj_mod[j, i] = 1

            elif op == "add":
                non_existing = [(i, j) for i in range(n) for j in range(n)
                                if i != j and adj_mod[i, j] == 0 and adj_mod[j, i] == 0]
                if non_existing:
                    idx = rng.integers(len(non_existing))
                    i, j = non_existing[idx]
                    adj_mod[i, j] = 1

        # Must be a DAG
        if _has_cycle(adj_mod):
            continue

        # Must be different from truth
        if np.array_equal(adj_mod, adj_true):
            continue

        # Must not duplicate existing candidate
        is_dup = any(np.array_equal(adj_mod, np.array(c["adjacency"])) for c in candidates)
        if is_dup:
            continue

        ci = len(candidates) - 1
        candidates.append({
            "id": f"G{len(candidates) + 1}",
            "confidence": float(round(confidences[ci], 3)) if ci < len(confidences) else 0.35,
            "description": f"Modified ({n_modifications} edge changes)",
            "adjacency": adj_mod.tolist(),
        })

    # Re-number
    for i, c in enumerate(candidates):
        c["id"] = f"G{i + 1}"

    return candidates


def generate_scenario(n: int, edges_per_var: float = 1.5,
                      n_candidates: int = None, n_modifications: int = None,
                      seed: int = 42, output_dir: str = "data") -> dict:
    """Generate complete scenario: ground truth + candidates.

    Difficulty scales with graph size:
    - K increases: more competitors
    - n_modifications increases: candidates differ by more edges
    """
    if n_candidates is None:
        n_candidates = min(6 + n, 20)
    if n_modifications is None:
        n_modifications = max(1, n // 5)

    variables, topo_order, adj, edges = random_dag(n, edges_per_var, seed)
    weights = assign_weights(edges, seed)

    edge_list = []
    for src, tgt in edges:
        w = weights[(src, tgt)]
        edge_list.append({
            "source": src, "target": tgt,
            "weight": w,
            "sign": "inhibitory" if w < 0 else "excitatory",
        })

    true_weights = {f"w_{src}{tgt}": w for (src, tgt), w in weights.items()}

    gt_json = {
        "description": f"Synthetic random DAG: {n} variables, {len(edges)} edges",
        "variables": {v: {"name": v} for v in variables},
        "topological_order": topo_order,
        "edges": edge_list,
        "adjacency_matrix": {"order": variables, "matrix": adj.tolist()},
        "scm_parameters": {"noise_variance": 0.3, "weights": true_weights},
    }

    candidates = generate_candidates(
        variables, adj, edges, n_candidates, n_modifications, seed
    )

    cand_json = {
        "description": f"Candidate graphs for {n}-variable synthetic DAG",
        "variable_order": variables,
        "candidates": [{
            "id": c["id"],
            "confidence": c["confidence"],
            "description": c["description"],
            "edges": [],
            "adjacency_matrix": c["adjacency"],
            "rationale": c["description"],
        } for c in candidates],
    }

    size_key = f"{n}var"
    out = Path(output_dir) / size_key
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "ground_truth_dag.json", "w") as f:
        json.dump(gt_json, f, indent=2)
    with open(out / "candidate_graphs.json", "w") as f:
        json.dump(cand_json, f, indent=2)

    info = {
        "size_key": size_key,
        "n_variables": n,
        "n_edges": len(edges),
        "n_candidates": len(candidates),
        "n_modifications": n_modifications,
        "true_weights": true_weights,
    }

    print(f"Generated {size_key}: {n} vars, {len(edges)} edges, "
          f"{len(candidates)} candidates (mods={n_modifications})")

    return info


SIZES = [5, 7, 10, 15, 20, 25, 30]


if __name__ == "__main__":
    for n in SIZES:
        generate_scenario(n, seed=42)
