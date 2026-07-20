# Paper Figures — Infra-Bench CLS

Standalone script to regenerate all figures in the Infra-Bench CLS
manuscript from per-seed and aggregate results JSONs.

## Requirements

```
pip install matplotlib numpy
```

## Configuration

Edit the top of `paper_figures.py`:

```python
RESULTS_DIR    = Path('./results')     # per-seed JSONs (one subdir per condition)
AGGREGATES_DIR = Path('./aggregates')  # aggregate JSONs (one per condition)
OUTPUTS_DIR    = Path('./figures')     # where PNGs are written
```

## Directory structure expected

```
results/
  fm_eval_dinov3_finetune_v1/
    dinov3_finetune_v1_seed314_results.json
    dinov3_finetune_v1_seed271_results.json
    dinov3_finetune_v1_seed161_results.json
  fm_eval_dinov3_lp_v1/
    ...
  (one subdir per condition)

aggregates/
  dinov3_finetune_v1_aggregate.json
  dinov3_lp_v1_aggregate.json
  ...
```

## Usage

```
# All figures
python paper_figures.py

# Single main-text figure
python paper_figures.py --figure 4

# Single appendix figure set
python paper_figures.py --appendix A
```

## Figures generated

| Figure | Description                                                      |
|--------|------------------------------------------------------------------|
| 3      | All 30 conditions sorted ascending by test macro F1              |
| 4      | Grouped-by-model horizontal bar chart                            |
| 5      | FT-only labels-efficiency overlay (1.0x vs 0.3x)                 |
| 6      | LP + FT training dynamics at 0.3x with baselines                 |
| 7      | LP + FT training dynamics at 1.0x with baselines                 |
| A1–A9  | Per-model aggregate confusion matrices                           |
| E1     | Per-sector macro F1, all models                                  |
| E2     | Per-region macro F1, all models                                  |

## Notes

- All figures use the canonical FM ordering defined at the top of the file
- Baselines (Supervised ResNet-18, Random Features) are visually distinct
  (dashed lines, gray/brown palette) in training-dynamics figures
- Confusion matrices use the Oranges palette and are row-normalized
- Best-checkpoint epoch is marked with a star on each training curve
