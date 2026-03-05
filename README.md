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

| Variables | ~Edges | K (candidates) | Mods per candidate |
|-----------|--------|----------------|--------------------|
| 5         | 8      | 11             | 1                  |
| 7         | 10     | 13             | 1                  |
| 10        | 15     | 16             | 2                  |
| 15        | 22     | 20             | 3                  |
| 20        | 30     | 20             | 4                  |
| 25        | 38     | 20             | 5                  |
| 30        | 45     | 20             | 6                  |

## Running

```bash
# Generate all data files
python -m src.synthetic_graphs

# Run experiment
python -m src.run_synthetic

# Plot results
python -m src.plot_synthetic logs/synthetic_results_<timestamp>.json
```
