# Results site

Published at <https://justing0909.github.io/infra-bench-cls>, served by GitHub
Pages from this folder on `master` (Settings → Pages → Source: `master` /
`/docs`).

Plain HTML/CSS/JS — no Jekyll, no framework, no build step. `.nojekyll` is
present so Pages serves the files as-is. Every path in the site is relative, so
the folder works unchanged at a project-page URL or moved to a domain root.

Note that the repository `.gitignore` excludes `figures/` at any depth, for the
regenerated paper figures. `!docs/figures/` re-includes the copies this site
ships; without it they are silently dropped from commits.

```
index.html          the filterable results table
figures.html        the five figures
notes.html          method notes, caveats, citation
style.css           dark earth palette, serif type
script.js           shared — results table + citation copy button
data/results.json   all 30 conditions — the single source of truth
figures/            5 figures exported from the paper pipeline
tools/              data generator + validator (not served)
README.md           this file (not served)
```

## Design

Three pages with a shared nav. `index.html` opens with a recommender, then
carries the full results table.

Below that is one filterable table rather than a stack of static slices: pick a
**breakdown** (overall / by class / by sector / by region) to change the
columns, a **metric** for the overall view, and filter by model, adaptation
protocol and data scale. Any heading re-sorts.

## The recommender

"Have only a couple of minutes?" turns four questions — compute budget, label
budget, target class or sector, region — into a model recommendation, scored
against the conditions the benchmark actually measured. It is not a heuristic:
it filters the 30 conditions and ranks what survives.

- **Compute** maps to real recorded numbers. The `nogpu` tier restricts to
  linear probes; `modest` applies a 12 GB ceiling against each fine-tune run's
  measured `peak_gpu_gb`, which drops CROMA FT (~40 GB) and Prithvi FT
  (~17–19 GB). Verified: the ceiling excludes exactly the four runs above it.
- **Labels** maps to data scale — thousands to 1.0×, a few hundred or less to
  0.3×, "not sure" keeps both in the running. The note covers the case the
  tiers cannot serve: with no labels at all, none of these results apply, since
  every condition here is supervised.

The first two questions carry a superscript **?** that opens a note: what VRAM
means and how to look yours up on Windows or macOS, and what actually counts as
a label along with the file formats you would hold them in. Only one note is
open at a time. The badge stays small to read as a superscript, so the hit area
is grown with a pseudo-element instead — capped at 8px vertically, because the
gap down to the segmented control is 9px and any more would swallow clicks
meant for it.
- **Classes** are chips, laid out one sector per row with that sector's classes
  alongside. Pick any combination; the ranking becomes the mean of the selected
  classes' F1 scores. This genuinely changes the answer: DINOv3 wins most
  classes, but OlmoEarth leads Solar Farm and SatlasPretrain S2 leads
  Distribution Substation. The sector button on each row toggles its whole set,
  so two sectors is two clicks, and shows a part-selected state when only some
  of its classes are on.
- **Regions** are chips too, but they **never** enter the ranking. There is no
  per-region, per-class figure anywhere in the results — per-region is on the
  13-class basis, per-class on the 10-class subset — so nothing can answer
  "best at airports in Asia". Regions are reported alongside instead: the
  pick's score across the chosen regions, plus the regional leader if it
  differs.

  This used to work differently: region drove the ranking when no classes were
  picked. That meant deselecting one sector silently swapped the whole basis
  and could change the winner for no visible reason. Consistency is worth more
  than using the input.

Selecting a full sector is not an approximation — it reproduces the paper's
per-sector F1 exactly, because that metric is defined the same way (the macro
average of member classes, each scored on the full test set). Same for all ten
classes and macro F1. Verified against the data: 120/120 sector and 30/30 macro
identities hold to 1e-9.

Both pickers start **empty** and you add what you want. Empty means "no
narrowing" — the ranking uses the whole-set metric. Clicking through every
sector to deselect ends at nothing selected rather than snapping back to all,
and there are explicit select-all / clear controls.

A **metric selector** covers the six whole-set metrics, macro F1 by default,
each with a one-line explanation. It is live whenever the selection is the
whole set — nothing picked *or* all ten picked, since both mean the same thing.
Only a proper subset (1–9 classes) locks it, because the stored results break
out F1 alone per class; it greys out and says why.

Scores are shown as percentages rather than three-decimal fractions. With a
single class selected the panel also states the two rates F1 is made of, both
recovered from the summed confusion matrix: the share of real sites of that
class the model finds, and how often it is right when it flags one. Power Plant
is the case that shows why this matters — F1 reads 35.0%, but the model finds
only 30% of real power plants and is right 45% of the time when it claims one.
Neither number is visible in "0.350".

Differences between two percentages are labeled in **points**, not percent.

Two more things it surfaces rather than hides. It flags when the gap to the
runner-up is **smaller than the two runs' combined seed spread**, so a
2-point lead over three seeds is not read as a real difference. And when the
winner is an expensive backbone (>300 GMACs) it names the best cheap
alternative with the accuracy cost attached — otherwise answering "no GPU"
would still return a 955-GMAC ViT-L with no caveat.

The seed-spread check is a rough guard, not a significance test; three seeds
will not support one.

"Show this in the table" applies the pick's protocol and scale, switches to the
breakdown that shows what was ranked, **and sorts by that same column** — one
selected class sorts by that class, a full sector by that sector, and so on.
Without this the recommended row could arrive sitting below rows that beat it
on a metric the reader never asked about.

The table's class and region columns follow the selection too, so its Mean
column is the same quantity the recommendation was ranked on. With everything
selected — the default — every column shows.

The recommended row is marked with a tan edge and a warm wash. The mark
persists through sorting and breakdown changes, and pulses once on arrival
(suppressed under `prefers-reduced-motion`).

`script.js` is shared across all three pages and guards on element presence,
so it is safe to include everywhere — it builds the table only where
`#t-results` exists, and wires the citation copy button only where a
`.copy` button exists.

Type is EB Garamond for headings, Crimson Pro for body, IBM Plex Mono for
labels and all numerics — matching
[jg-geoportfolio](https://jg-geoportfolio.vercel.app). The palette is dark and
earthy, sharing that site's `#1a1e1b` background. Dark only; there is no light
theme.

Everything sits in one 848 px centered column, whose width is set by the
recommendation panel — no block gets its own narrower measure, so section
rules, tables, prose and figures all break on the same edges. Headings and
intros are centered; body copy, tables and lists stay left-aligned.

Body type is 18.5 px. The column was widened alongside that bump so the line
measure stayed roughly constant rather than shortening.

Every page header is identical by construction: eyebrow, title, lede, optional
aside, then the nav. On subpages the eyebrow links home. The `h1` lands at the
same y-offset on all three.

Sections are separated mostly by space — 96 px above, 58 px below the rule —
with a warm hairline at `rgba(200,176,138,.13)`, deliberately dimmer than the
table rules so it reads as a division rather than another row.

Two palette colors (`#7a5c3d` brown, `#2f4a3a` green) sit at roughly 2.8:1 on
this background, which is too low for the small mono labels, so `--brown-lit`
and `--green-lit` are lifted variants used wherever those colors carry text.
All text on the site clears WCAG AA at 4.8:1 or better.

## Citation

`notes.html#cite` carries a BibTeX block with a copy button. It is currently a
`@misc` entry pointing at the GitHub repository, because the paper is not yet
posted — **swap it for the paper entry once that is on arXiv**, and add the
dataset's Zenodo DOI alongside when one is minted.

`notes.html#team` carries the author list with affiliations, every name linked.
The front page header links straight to it.

## The data file

`data/results.json` holds the **10-class** results: 9 models across up to
4 conditions each (2 adaptation protocols × 2 data scales), 3 seeds per
condition, 30 conditions total. Per condition it carries six aggregate metrics
plus per-class, per-sector, and per-region breakdowns, a summed 10×10 confusion
matrix, and GMACs/params.

The site does not currently render the confusion matrices, though the data for
them is in the file — add a breakdown to `script.js` if you want them.

It is generated from the evaluation run artifacts — the 30 `*_aggregate.json`
files and the 90 per-seed `*_results.json` confusion matrices produced by the
notebooks in the `infra-bench-cls` repo. The generator applies the same
10-class re-fit as `plots/paper_figures.ipynb` (excluding wind farm, port
terminal, and water works from the 13-class ontology) so every number here
matches the manuscript.

To refresh the data, point the generator at a copy of the results tree:

```bash
python tools/build_results.py /path/to/results data/results.json
python tools/validate.py data/results.json
```

`validate.py` checks the output against every figure printed in
`paper_figures.ipynb` — the Supporting Information Tables S10–S13 (all four
metrics, mean and std,
30 conditions), the LP→FT macro F1 values at full float precision, and the
structural constants. It exits non-zero on any mismatch, so a silent drift in
the upstream results will not reach the site unnoticed. Nothing else needs to
change.

## Two things to know when reading the tables

**Per-region F1 is 13-class.** The stored results record per-region F1 already
averaged over classes, so it cannot be recomputed on the 10-class subset.
That breakdown is on a different basis from everything else on the site and
raises a callout whenever it is selected. Compare models against each other
within it, not against numbers elsewhere.

**Some GMACs values are architecture proxies**, marked with `†`. Where the
original loader was unavailable, an architecture-matched stand-in was profiled
instead; GMACs depends on architecture rather than on trained weights.

## Local preview

`fetch()` will not read `data/results.json` over `file://`, so serve the folder:

```bash
python -m http.server 8899
```

Then open <http://localhost:8899>.
