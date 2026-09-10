# Project page

Published at <https://justing0909.github.io/infra-bench-cls>, served by GitHub
Pages from this folder on `master` (Settings → Pages → Source: `master` /
`/docs`).

A single page: a brief on the benchmark, one headline figure, links to the
paper, dataset and code, and BibTeX for the paper and the dataset. Plain
HTML/CSS/JS, no Jekyll, no framework, no build step. `.nojekyll` is present so
Pages serves the files as-is. Every path is relative, so the folder works at a
project-page URL or moved to a domain root.

```
index.html          the whole site
style.css           dark earth palette, serif type
script.js           citation copy buttons + margin sketches
figures/            fig_perclass_heatmap.png is the one the page shows
figures/sketches/   five hand-drawn margin decorations
data/, tools/       not served; see below
README.md           this file (not served)
```

It used to be three pages with an interactive results explorer. That was cut
back to a brief plus links, on the view that the paper carries the results and
the site should point at it. The explorer, the figures page and the method notes
are all in git history if any of it is wanted back.

## Files kept but not served

`figures/` still holds four figures the page no longer shows (`fig3`, `fig4`,
`fig6`, `fig7`). They are kept because the repository `.gitignore` excludes
`figures/` at any depth for the regenerated paper figures, so these copies under
`docs/` are the only ones committed anywhere — deleting them would lose them
from the repo rather than merely from the site.

`data/results.json` and `tools/` are likewise unused by the page now. They are
the generator and validator for the numbers: `tools/build_results.py` rebuilds
`results.json` from the evaluation run artifacts applying the same 10-class
re-fit as `plots/paper_figures.ipynb`, and `tools/validate.py` checks it against
every figure printed in that notebook, exiting non-zero on a mismatch. Worth
keeping even unused.

Note that `!docs/figures/` in the repository `.gitignore` is what re-includes
this folder's images. Without it they are dropped from commits silently.

## Editing

The five margin sketches are pinned to fixed slots in `SKETCH_SLOTS` at the
bottom of `script.js`; edit the `img` field to rearrange. They only render from
1280px up, and are only created when that breakpoint matches, so a narrow screen
issues no requests for them.
