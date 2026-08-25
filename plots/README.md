# Paper figures

[`paper_figures.ipynb`](paper_figures.ipynb) regenerates every figure in the
manuscript. It is the authoritative source for them, and there is no local
script equivalent.

**It runs in Google Colab.** The first cell mounts Drive; everything reads from
the evaluation output tree at `MyDrive/infra_fm/results/`. See
[Running the notebooks](../README.md#running-the-notebooks) in the root README
for how to open it, either in the browser or from VS Code.

## Configuration

One constant, set in the config cell:

```python
RESULTS_BASE = Path('/content/drive/MyDrive/infra_fm/results')
OUTPUTS_DIR  = RESULTS_BASE / 'figures'
```

`RESULTS_BASE` holds one subdirectory per experimental condition, each with
three per-seed JSONs and one aggregate:

```
results/
  fm_eval_dinov3_finetune_v1/
    dinov3_finetune_v1_seed314_results.json
    dinov3_finetune_v1_seed271_results.json
    dinov3_finetune_v1_seed161_results.json
    dinov3_finetune_v1_aggregate.json
  fm_eval_dinov3_lp_v1/
  ...
```

`RUN_DIR_MAP` in the config cell maps each model to its four condition
directories in the order LP 1.0x, LP 0.3x, FT 1.0x, FT 0.3x, with `None` where
a model has no such condition. AlphaEarth is linear-probe only; the supervised
ResNet-18 has no probe.

## How to run it

Run the setup cells once, in order: Drive mount, results-tree listing, config
and loaders, paper style, and the 10-class re-fit. Then run any figure cell on
its own. Each renders inline and overwrites its PNG.

The 10-class re-fit cell must run before any figure cell. It monkey-patches
`load_agg()` and `_best_condition_agg()`, so a figure cell run before it will
silently produce 13-class numbers. The cell prints a sanity check against four
known values and reports `OK` or `MISMATCH` for each.

## Figures

Cells run in manuscript order. Function names and output filenames both match
the paper, so the PNGs drop straight into the manuscript's `figures/` folder
without renaming.

### Main text

| Figure | Function | Output file |
|---|---|---|
| 3 — per-class F1 heatmap | `figure_3_perclass_heatmap()` | `fig3_perclass_heatmap_across_f1s.png` |
| 4 — all 30 conditions, sorted | `figure_4_all_conditions()` | `fig4_all_conditions.png` |
| 5 — LP to FT flip | inline in the flip cell | `fig5_lp_to_ft_flip.png` |
| 6 — grouped by model | `figure_6_grouped_by_model()` | `fig6_grouped_by_model.png` |
| 7 — fine-tuning labels efficiency | `figure_7_labels_efficiency_ft()` | `fig7_fine-tuning_labels-efficiency.png` |
| 8 — per-sector macro F1 | `figure_8_per_sector()` | `fig8_per_sector.png` |
| 9 — per-region macro F1 | `figure_9_per_region()` | `fig9_per_region.png` |
| 10 — training dynamics, 1.0x | `figure_10_training_dynamics_10x()` | `fig10_training-dynamics_1.0x.png` |

Figures 1 and 2, the curation and evaluation pipeline diagrams, are drawn by
hand and are not produced here.

### Supporting Information

| Figure | Function | Output file |
|---|---|---|
| S1–S9 — confusion matrices | `save_all_confusion_matrices_10class()` | `FigS1_satlas_s1.png` … `FigS9_random_features.png` |
| S10 — training dynamics, 0.3x | `figure_s10_training_dynamics_03x()` | `FigS10_training-dynamics_0.3x.png` |

`_SI_FIGURE_NAMES` in the confusion-matrix cell maps each model key to its
output stem. Two of them differ from the model key, matching the paper:
`olmoearth` writes `FigS6_olmo` and `dinov3` writes `FigS7_dino`.

Tables S10–S13 are printed rather than drawn. `print_si_metric_tables()`
recomputes accuracy, macro precision, macro recall, and weighted precision on
the 10-class subset from the per-seed confusion matrices. The remaining
Supporting Information tables are built elsewhere: S5 and S6, the wall-clock
timings, come from
[`evaluation/analysis/confusion_matrices.ipynb`](../evaluation/analysis/confusion_matrices.ipynb),
and S7, the ontology definitions, from [`ONTOLOGY.md`](../ONTOLOGY.md).

## Conventions

- Models appear in the `FM_ORDER` defined in the config cell, with a fixed
  color per model in `FM_COLORS`. Both are used by every figure and by the
  results site, so changing one means changing the other.
- The two baselines, supervised ResNet-18 and random features, are drawn
  dashed in grey and brown so they read as baselines in the training-dynamics
  panels.
- Confusion matrices are row-normalized and use the Oranges palette.
- Each training curve marks its best-validation epoch with a star, which is the
  checkpoint restored for the test pass.

## Taxonomy

The aggregate JSONs on Drive carry 13-class metrics. The re-fit cell recomputes
macro F1, weighted F1, and per-sector F1 on the 10 classes the paper reports,
dropping wind farm, water works, and port terminal. Per-class F1 is unaffected,
since a class's own F1 does not depend on which other classes are averaged
with it.

Per-region F1 is the exception and stays 13-class. The aggregates store it
already averaged over classes, so there is nothing left to recompute from, and
the checkpoints needed to redo it were not kept. The manuscript states this in
the Figure 9 caption and in the limitations.

The same re-fit is ported into [`docs/tools/build_results.py`](../docs/tools/build_results.py),
and [`docs/tools/validate.py`](../docs/tools/validate.py) checks the site's
numbers against what this notebook prints.
