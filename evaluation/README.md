# Evaluation

Thirty conditions: seven backbone configurations plus two baselines, each at
1.0x and 0.3x training data, under linear probing and fine-tuning where the
model supports both. One notebook per condition.

Every notebook runs in Google Colab on a GPU runtime and reads the dataset from
Drive. See [Running the notebooks](../README.md#running-the-notebooks) for the
Drive layout and how to open them.

```
alphaearth/   lp_{1.0x,0.3x}                      precomputed 64-D embeddings, LP only
croma/        lp_{1.0x,0.3x}, ft_{1.0x,0.3x}      S1 + S2 joint
dinov3/       lp_{1.0x,0.3x}, ft_{1.0x,0.3x}      RGB, non-EO baseline model
olmoearth/    lp_{1.0x,0.3x}, ft_{1.0x,0.3x}      S2
prithvi/      lp_{1.0x,0.3x}, ft_{1.0x,0.3x}      S2
satlas_s1/    lp_{1.0x,0.3x}, ft_{1.0x,0.3x}      S1
satlas_s2/    lp_{1.0x,0.3x}, ft_{1.0x,0.3x}      S2
resnet18/     supervised_{1.0x,0.3x}              fully trained baseline
              random_features_lp_{1.0x,0.3x}      frozen random init baseline
analysis/     cross-model utilities
```

## Running one condition

1. Open the notebook in Colab on a GPU runtime. Linear probes will run without
   one; fine-tuning will not.
2. Check that `MyDrive/infra_fm/datasets/` holds the 28 cells. The split
   artifact and the AlphaEarth embeddings come from the code zip, so there is
   nothing else to stage.
3. Set `SMOKE_ONLY = False` in the training-cell parameters.
4. Run all cells. Per-seed results go to `results/fm_eval_<name>/`, and the
   aggregate is written once all three seeds are present.

Fine-tuning notebooks resume from `checkpoint_final.pt` if Colab disconnects
part way through a seed.

Every aggregate write is guarded so a partial rerun cannot overwrite a
complete one. Thirty of them test `set(SEEDS) == set(FULL_PROTOCOL_SEEDS)` and
print a note instead of writing when it fails. The four combine-from-disk cells
in the SatlasPretrain fine-tune notebooks raise `FileNotFoundError` unless all
three per-seed JSONs are on disk. Each aggregate also stamps `agg['seeds']`.

## Environments

Each notebook installs its own dependencies, because the foundation-model
stacks conflict with one another. Nothing FM-specific belongs in the top-level
`requirements.txt`.

The published runs were not fully pinned. What each notebook pins today:

| Model | Pinned | Unpinned |
|---|---|---|
| Prithvi-EO-2.0 | `terratorch==0.99.8`, `transformers==4.41.0`, `huggingface-hub==0.36.2`, `torch==2.6.0`, `torchvision==0.21.0`, `pillow<12` | `pyarrow`, `scikit-learn`, `scipy` |
| OlmoEarth | `huggingface-hub==0.36.2` | `olmoearth_pretrain_minimal`, `hf_transfer`, `pyarrow`, `scikit-learn`, `scipy` |
| DINOv3 | `transformers>=4.49.0,<5.0`, `numpy>=2.2` | `huggingface-hub`, `pyarrow`, `scikit-learn`, `scipy` |
| SatlasPretrain S1, S2 | — | `satlaspretrain-models`, `scikit-learn`, `pyarrow` |
| CROMA | — | `einops`, `huggingface_hub`, `scikit-learn`, `pyproj`, `pyarrow` |
| AlphaEarth | — | `pyarrow`, `pyproj`, `scikit-learn` |
| ResNet-18 baselines | — | `scikit-learn`, `pyarrow` |

The exact resolved versions from the published runs are not recoverable: the
install cells use `%%capture` or `-q`, so pip's output was not kept in the
committed notebooks. Treat the pinned entries above as the reproducibility
floor and the rest as "whatever Colab resolved in mid-2026".

Before a rerun you intend to cite, capture the environment so the next person
does not face the same gap. Add a cell after the installs:

```python
import subprocess, json, pathlib
freeze = subprocess.run(['pip', 'freeze'], capture_output=True, text=True).stdout
out = pathlib.Path(OUTPUT_DIR) / 'pip_freeze.txt'
out.write_text(freeze)
print(f'environment recorded -> {out}  ({len(freeze.splitlines())} packages)')
```

`OUTPUT_DIR` already points at that condition's results folder, so the freeze
lands beside the per-seed JSONs and travels with them.

## Model weights

Weights download from each model's original source at run time, mostly via
`hf_hub_download`. No notebook sets `HF_HOME`, `HF_HUB_CACHE`, or
`HF_HUB_OFFLINE`, so nothing has to be pre-staged — a fresh runtime fetches
what it needs.

A few notebooks print "Fine when HF_HUB_OFFLINE=1 and models are pre-loaded in
Drive cache." That describes a setup used during development; it is not what
the committed notebooks do.

Token handling: OlmoEarth reads `HF_TOKEN_OLMOEARTH`, others `HF_TOKEN`. The
chain is Colab Secrets, then `.env` on Drive, then the ambient environment. The
token is only needed for gated metadata calls, so "no token found" is not
automatically a failure.

Because weights are pulled live, an upstream checkpoint being withdrawn or
silently updated is the one dependency outside this repository's control.

## Cross-model utilities

| Notebook | Purpose |
|---|---|
| `spatial_split_verification.ipynb` | builds `asset_id_to_split_v1.parquet`; gated on its own verification checks |
| `confusion_matrices.ipynb` | per-seed confusion matrices and the wall-clock backfill behind Tables S5 and S6 |
| `compute_weighted_f1.ipynb` | weighted F1 recomputation, cross-checked against a private `claims.json` of hand-recorded numbers. **Author-only** — that file is not in the repository or the dataset, so this notebook will not run elsewhere. `docs/tools/validate.py` covers the same ground publicly |
| `per_sector_f1_catchall.ipynb` | the corrected per-sector F1 definition |
| `find_GMACs.ipynb` | GMACs profiling; uses timm architecture proxies where a model's own loader was unavailable, which is what the `†` marks on the results site refer to |

## Known quirks

**SatlasS1 and SatlasS2 cannot share a runtime.** Both use
`/content/datasets/` and race on the extract step. Run them separately.

**Prithvi's loader is fragile.** It needs `trust_remote_code=True` and
`num_labels=0`; the TerraTorch scipy/dask/rapids conflict is documented in the
notebook itself.

**Large parquets exhaust Colab RAM.** Read only the columns you need:

```python
df = pd.read_parquet(path)[['asset_id', 'asset_type', 'lat', 'lon']].copy()
```
