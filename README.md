# CBO Synthetic Scaling Experiment

Tests CBO graph recovery on purely synthetic random DAGs from 5 to 30 variables.
Isolates the scaling question from biological grounding.

## Design

- **Random DAGs** with fixed edge density (1.5 edges per variable)
- **Weights** drawn from [-0.8, -0.3] ∪ [+0.3, +0.8]
- **Difficulty scales with size**:
  - K (candidates) = min(6 + n, 20)
  - Modification depth = max(1, n/5) edge changes per candidate
- **Multiple seeds** for error bars

## Sizes Tested

| Variables | ~Edges | K (candidates) | Mods | Setting |
|-----------|--------|----------------|------|---------|
| 5         | 8      | 10             | 1    | Single-edge modifications |
| 7         | 10     | 14             | 1    | Single-edge modifications |
| 10        | 15     | 20             | 1    | Single-edge modifications |
| 15        | 22     | 25             | 1    | Single-edge modifications |
| 20        | 30     | 25             | 1    | Single-edge modifications |
| 25        | 38     | 25             | 1    | Single-edge modifications |
| 30        | 45     | 25             | 1    | Single-edge modifications |

All candidates differ from ground truth by exactly one edge (removal, reversal, or addition),
making them maximally confusable. K scales as min(2n, 25).

## Running

```bash
# Generate all data files
python -m src.synthetic_graphs

# Run experiment
python -m src.run_synthetic

# Plot results
python -m src.plot_synthetic logs/synthetic_results_<timestamp>.json
```
