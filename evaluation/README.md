# Evaluation

Thirty conditions are evaluated here, covering seven backbone configurations
plus two baselines, each at 1.0x and 0.3x training data and under both linear
probing and fine-tuning wherever the model supports them, with one notebook per
condition. All of them run in Google Colab on a GPU runtime and read the
dataset from Drive, so the Drive layout and the two ways of opening a notebook
are described in
[Running the notebooks](../README.md#running-the-notebooks).

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

1. Open the notebook in Colab on a GPU runtime, which fine-tuning requires
   and linear probing does not.
2. Check that `MyDrive/infra_fm/datasets/` holds the 28 cells, which is all
   that needs staging, since the split artifact and the AlphaEarth embeddings
   arrive with the code zip.
3. Set `SMOKE_ONLY = False` in the training-cell parameters.
4. Run all cells. Per-seed results go to `results/fm_eval_<name>/`, and the
   aggregate is written once all three seeds are present.

Should Colab disconnect part way through a seed, the fine-tuning notebooks
resume from `checkpoint_final.pt` rather than starting the seed again.

Every aggregate write is guarded so that a partial rerun cannot overwrite a
complete one, in one of two ways. Thirty of them test
`set(SEEDS) == set(FULL_PROTOCOL_SEEDS)` and print a note instead of writing
when that fails, while the four combine-from-disk cells in the SatlasPretrain
fine-tune notebooks raise `FileNotFoundError` unless all three per-seed JSONs
are already on disk. Each aggregate additionally stamps `agg['seeds']`, so a
file that did slip through would at least describe itself.

## Environments

Each notebook installs its own dependencies, because the foundation-model
stacks conflict with one another and so nothing model-specific belongs in the
top-level `requirements.txt`. The published runs were not fully pinned, and
what each notebook pins today varies considerably:

| Model | Pinned | Unpinned |
|---|---|---|
| Prithvi-EO-2.0 | `terratorch==0.99.8`, `transformers==4.41.0`, `huggingface-hub==0.36.2`, `torch==2.6.0`, `torchvision==0.21.0`, `pillow<12` | `pyarrow`, `scikit-learn`, `scipy` |
| OlmoEarth | `huggingface-hub==0.36.2` | `olmoearth_pretrain_minimal`, `hf_transfer`, `pyarrow`, `scikit-learn`, `scipy` |
| DINOv3 | `transformers>=4.49.0,<5.0`, `numpy>=2.2` | `huggingface-hub`, `pyarrow`, `scikit-learn`, `scipy` |
| SatlasPretrain S1, S2 | none | `satlaspretrain-models`, `scikit-learn`, `pyarrow` |
| CROMA | none | `einops`, `huggingface_hub`, `scikit-learn`, `pyproj`, `pyarrow` |
| AlphaEarth | none | `pyarrow`, `pyproj`, `scikit-learn` |
| ResNet-18 baselines | none | `scikit-learn`, `pyarrow` |

The exact resolved versions from the published runs are not recoverable: the
install cells use `%%capture` or `-q`, so pip's output was not kept in the
committed notebooks. Treat the pinned entries above as the reproducibility
floor and the rest as "whatever Colab resolved in mid-2026".

Anyone rerunning a condition they intend to cite should capture the
environment first, so that the next person does not face the same gap, which
takes one cell placed after the installs:

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

Weights download from each model's original source at run time, mostly through
`hf_hub_download`, and because no notebook sets `HF_HOME`, `HF_HUB_CACHE`, or
`HF_HUB_OFFLINE`, a fresh runtime simply fetches what it needs and nothing has
to be pre-staged. A few notebooks do print "Fine when HF_HUB_OFFLINE=1 and
models are pre-loaded in Drive cache", which describes a setup used during
development rather than anything the committed notebooks do.

Tokens are read from Colab Secrets first, then from `.env` on Drive, and
finally from the ambient environment, with OlmoEarth looking for
`HF_TOKEN_OLMOEARTH` and the rest for `HF_TOKEN`. Since a token is only needed
for gated metadata calls, a message reporting that none was found is not
automatically a failure. Pulling weights live does leave one dependency outside
this repository's control, in that an upstream checkpoint could be withdrawn or
silently updated between runs.

## Cross-model utilities

| Notebook | Purpose |
|---|---|
| `spatial_split_verification.ipynb` | builds `asset_id_to_split_v1.parquet`, gated on its own verification checks |
| `confusion_matrices.ipynb` | per-seed confusion matrices and the wall-clock backfill behind Tables S5 and S6 |
| `compute_weighted_f1.ipynb` | recomputes weighted F1 from the per-seed confusion matrices. it used to diff the result against a private `claims.json`, but that comparison was removed and the dead load with it, so it now runs anywhere a results tree does |
| `per_sector_f1_catchall.ipynb` | the corrected per-sector F1 definition |
| `find_GMACs.ipynb` | GMACs profiling, using timm architecture proxies where a model's own loader was unavailable, which is what the `†` marks on the results site refer to |

## Known quirks

SatlasS1 and SatlasS2 cannot share a runtime, because both use
`/content/datasets/` and race on the extract step, so they have to be run
separately. Prithvi's loader is similarly particular, needing
`trust_remote_code=True` together with `num_labels=0`, and its TerraTorch
conflict over scipy, dask and rapids is documented in that notebook.

Loading a full parquet will exhaust Colab RAM on the larger regions, so read
only the columns actually needed:

```python
df = pd.read_parquet(path)[['asset_id', 'asset_type', 'lat', 'lon']].copy()
```
