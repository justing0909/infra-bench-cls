"""
paper_figures.py

Regenerates all figures for the Infra-Bench CLS manuscript.

Each figure reads from per-seed and/or aggregate results JSONs and writes
a PNG to OUTPUTS_DIR.

Figures generated (matches manuscript numbering):
- Figure 3:  sorted bar chart across all 30 conditions
- Figure 4:  grouped-by-model horizontal bar chart
- Figure 5:  FT-only labels-efficiency overlay (1.0x solid vs 0.3x dashed)
- Figure 6:  0.3x training dynamics (LP + FT panels, with baselines)
- Figure 7:  1.0x training dynamics (LP + FT panels, with baselines)
- Appendix A (Figures A1-A9): aggregate confusion matrices per model
- Appendix E1: per-sector performance across all models
- Appendix E2: per-region performance across all models

Usage
-----
    # Set RESULTS_DIR and AGGREGATES_DIR at top of file, then:
    python paper_figures.py                  # generate all
    python paper_figures.py --figure 4       # generate a specific figure
    python paper_figures.py --appendix A     # generate an appendix figure set
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# =========================================================================
# CONFIGURATION - update these paths for your environment
# =========================================================================
RESULTS_DIR = Path('./results')          # per-seed JSONs, one subdir per condition
AGGREGATES_DIR = Path('./aggregates')    # per-condition aggregate JSONs
OUTPUTS_DIR = Path('./figures')          # where PNGs are written

# Canonical FM ordering used throughout the paper
FM_ORDER = [
    'SatlasPretrain S1',
    'SatlasPretrain S2',
    'CROMA',
    'Prithvi-EO-2.0',
    'AlphaEarth Foundations',
    'OlmoEarth v1.1-Base',
    'DINOv3 ViT-L/16',
    'Supervised ResNet-18',
    'Random Features',
]

# Canonical colors per model
FM_COLORS = {
    'SatlasPretrain S1':      '#e67e22',   # orange
    'SatlasPretrain S2':      '#1f77b4',   # blue
    'CROMA':                  '#e6b800',   # yellow-gold
    'Prithvi-EO-2.0':         '#9467bd',   # purple
    'AlphaEarth Foundations': '#2ca02c',   # green
    'OlmoEarth v1.1-Base':    '#17becf',   # cyan
    'DINOv3 ViT-L/16':        '#d62728',   # red
    'Supervised ResNet-18':   '#4d4d4d',   # dark gray
    'Random Features':        '#8B4513',   # brown
}

# 13-class labels for the benchmark
CLASS_LABELS = [
    "TX substation", "DX substation", "DX (other)",
    "Power plant", "Solar farm", "Wind farm",
    "Wastewater plant", "Water works", "Storage tank",
    "Airport", "Train station", "Port terminal",
    "Data center"
]

# Class support counts on the test set (n=2813 per seed)
CLASS_COUNTS = [105, 141, 511, 70, 98, 3, 228, 118, 574, 125, 737, 3, 100]

# =========================================================================
# CONFIG MAP: per-condition JSON paths
# Update these to point at your actual filenames.
# =========================================================================

# Each key is (model, mode, scale) -> aggregate JSON filename
AGGREGATE_MAP = {
    ('SatlasPretrain S1',    'LP', '1.0x'):        'satlas_s1_v2_aggregate.json',
    ('SatlasPretrain S1',    'LP', '0.3x'):        'satlas_s1_lp_0_3x_v1_aggregate.json',
    ('SatlasPretrain S1',    'FT', '1.0x'):        'satlas_s1_finetune_v1_aggregate.json',
    ('SatlasPretrain S1',    'FT', '0.3x'):        'satlas_s1_finetune_0_3x_v1_aggregate.json',

    ('SatlasPretrain S2',    'LP', '1.0x'):        'satlas_s2_v2_aggregate.json',
    ('SatlasPretrain S2',    'LP', '0.3x'):        'satlas_s2_lp_0_3x_v1_aggregate.json',
    ('SatlasPretrain S2',    'FT', '1.0x'):        'satlas_s2_finetune_v1_aggregate.json',
    ('SatlasPretrain S2',    'FT', '0.3x'):        'satlas_s2_finetune_0_3x_v1_aggregate.json',

    ('CROMA',                'LP', '1.0x'):        'croma_v2_aggregate.json',
    ('CROMA',                'LP', '0.3x'):        'croma_lp_0_3x_v1_aggregate.json',
    ('CROMA',                'FT', '1.0x'):        'croma_finetune_v1_aggregate.json',
    ('CROMA',                'FT', '0.3x'):        'croma_finetune_0_3x_v1_aggregate.json',

    ('Prithvi-EO-2.0',       'LP', '1.0x'):        'prithvi_v2_aggregate.json',
    ('Prithvi-EO-2.0',       'LP', '0.3x'):        'prithvi_lp_0_3x_v1_aggregate.json',
    ('Prithvi-EO-2.0',       'FT', '1.0x'):        'prithvi_finetune_v1_aggregate.json',
    ('Prithvi-EO-2.0',       'FT', '0.3x'):        'prithvi_finetune_0_3x_v1_aggregate.json',

    ('AlphaEarth Foundations', 'LP', '1.0x'):      'alphaearth_v2_aggregate.json',
    ('AlphaEarth Foundations', 'LP', '0.3x'):      'alphaearth_lp_0_3x_v1_aggregate.json',

    ('OlmoEarth v1.1-Base',  'LP', '1.0x'):        'olmoearth_lp_v1_aggregate.json',
    ('OlmoEarth v1.1-Base',  'LP', '0.3x'):        'olmoearth_lp_0_3x_v1_aggregate.json',
    ('OlmoEarth v1.1-Base',  'FT', '1.0x'):        'olmoearth_finetune_v1_aggregate.json',
    ('OlmoEarth v1.1-Base',  'FT', '0.3x'):        'olmoearth_finetune_0_3x_v1_aggregate.json',

    ('DINOv3 ViT-L/16',      'LP', '1.0x'):        'dinov3_lp_v1_aggregate.json',
    ('DINOv3 ViT-L/16',      'LP', '0.3x'):        'dinov3_lp_0_3x_v1_aggregate.json',
    ('DINOv3 ViT-L/16',      'FT', '1.0x'):        'dinov3_finetune_v1_aggregate.json',
    ('DINOv3 ViT-L/16',      'FT', '0.3x'):        'dinov3_finetune_0_3x_v1_aggregate.json',

    ('Supervised ResNet-18', 'Sup', '1.0x'):       'resnet18_supervised_v1_aggregate.json',
    ('Supervised ResNet-18', 'Sup', '0.3x'):       'resnet18_supervised_0_3x_v1_aggregate.json',

    ('Random Features',      'LP', '1.0x'):        'resnet18_random_features_v1_aggregate.json',
    ('Random Features',      'LP', '0.3x'):        'resnet18_random_features_0_3x_v1_aggregate.json',
}

# Per-seed run subdirectories (for training-dynamics figures and confusion matrices)
# Each value maps to (LP_1.0x_subdir, LP_0.3x_subdir, FT_1.0x_subdir, FT_0.3x_subdir)
RUN_DIR_MAP = {
    'SatlasPretrain S1':    ('fm_eval_satlas_s1_v2',                 'fm_eval_satlas_s1_lp_0.3x_v1',
                             'fm_eval_satlas_s1_finetune_v1',        'fm_eval_satlas_s1_finetune_0.3x_v1'),
    'SatlasPretrain S2':    ('fm_eval_satlas_s2_v2',                 'fm_eval_satlas_s2_lp_0.3x_v1',
                             'fm_eval_satlas_s2_finetune_v1',        'fm_eval_satlas_s2_finetune_0.3x_v1'),
    'CROMA':                ('fm_eval_croma_v2_spatial',             'fm_eval_croma_lp_0.3x_v1',
                             'fm_eval_croma_finetune_v1',            'fm_eval_croma_finetune_0.3x_v1'),
    'Prithvi-EO-2.0':       ('fm_eval_prithvi_v2_spatial',           'fm_eval_prithvi_lp_0.3x_v1',
                             'fm_eval_prithvi_finetune_v1',          'fm_eval_prithvi_finetune_0.3x_v1'),
    'AlphaEarth Foundations': ('fm_eval_alphaearth_v2_spatial',      'fm_eval_alphaearth_lp_0.3x_v1',
                               None,                                  None),
    'OlmoEarth v1.1-Base':  ('fm_eval_olmoearth_lp_v1',              'fm_eval_olmoearth_lp_0.3x_v1',
                             'fm_eval_olmoearth_finetune_v1',        'fm_eval_olmoearth_finetune_0.3x_v1'),
    'DINOv3 ViT-L/16':      ('fm_eval_dinov3_lp_v1',                 'fm_eval_dinov3_lp_0.3x_v1',
                             'fm_eval_dinov3_finetune_v1',           'fm_eval_dinov3_finetune_0.3x_v1'),
    'Supervised ResNet-18': (None, None,
                             'fm_eval_resnet18_supervised_v1',       'fm_eval_resnet18_supervised_0.3x_v1'),
    'Random Features':      ('fm_eval_resnet18_random_features_v1',  'fm_eval_resnet18_random_features_0.3x_v1',
                             None,                                    None),
}


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
def load_agg(model: str, mode: str, scale: str) -> Optional[dict]:
    """Load an aggregate JSON for a given condition."""
    key = (model, mode, scale)
    if key not in AGGREGATE_MAP:
        return None
    path = AGGREGATES_DIR / AGGREGATE_MAP[key]
    if not path.exists():
        print(f"  ! missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_val_curves(subdir: Optional[str]) -> Optional[np.ndarray]:
    """Load per-epoch val_macro_f1 across seeds for a given run subdir.

    Returns array of shape (n_seeds, n_epochs), or None if no data found.
    """
    if subdir is None:
        return None
    files = sorted(glob.glob(str(RESULTS_DIR / subdir / '*_seed*_results.json')))
    if not files:
        return None
    curves = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        inner = d.get('finetune') or d.get('linear_probe')
        if inner is None:
            continue
        hist = inner.get('history', [])
        vals = [h.get('val_macro_f1', 0.0) for h in hist]
        if vals:
            curves.append(vals)
    if not curves:
        return None
    min_len = min(len(c) for c in curves)
    return np.array([c[:min_len] for c in curves])


def load_best_epoch_mean(subdir: Optional[str]) -> Optional[float]:
    """Mean best-checkpoint epoch across seeds."""
    if subdir is None:
        return None
    files = sorted(glob.glob(str(RESULTS_DIR / subdir / '*_seed*_results.json')))
    if not files:
        return None
    epochs = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        inner = d.get('finetune') or d.get('linear_probe')
        if inner is not None and inner.get('best_epoch') is not None:
            epochs.append(inner['best_epoch'])
    return float(np.mean(epochs)) if epochs else None


def load_summed_confusion_matrix(subdir: str) -> Optional[np.ndarray]:
    """Sum per-seed confusion matrices for aggregate view."""
    files = sorted(glob.glob(str(RESULTS_DIR / subdir / '*_seed*_results.json')))
    if not files:
        return None
    cms = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        inner = d.get('finetune') or d.get('linear_probe')
        cm = inner.get('test', {}).get('confusion') if inner else None
        if cm is not None:
            cms.append(np.array(cm))
    return np.sum(cms, axis=0) if cms else None


# =========================================================================
# FIGURE 3: sorted bar chart across all 30 conditions
# =========================================================================
def figure_3_sorted_conditions():
    """All 30 conditions sorted ascending by test macro F1."""
    entries = []
    for (model, mode, scale), _ in AGGREGATE_MAP.items():
        d = load_agg(model, mode, scale)
        if d is None:
            continue
        entries.append({
            'model': model, 'mode': mode, 'scale': scale,
            'mean': d['test_macro_f1']['mean'],
            'std':  d['test_macro_f1']['std'],
        })
    entries.sort(key=lambda x: x['mean'])

    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(entries))
    colors = [FM_COLORS[e['model']] for e in entries]
    labels = [f"{e['model']} - {e['mode']} {e['scale']}" for e in entries]
    values = [e['mean'] for e in entries]
    stds = [e['std'] for e in entries]

    ax.barh(y, values, xerr=stds, color=colors, edgecolor='black', linewidth=0.5,
            error_kw={'ecolor': 'black', 'capsize': 2, 'linewidth': 0.7})
    for i, (v, s) in enumerate(zip(values, stds)):
        ax.text(v + max(s, 0.005) + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Test macro F1', fontsize=13)
    ax.set_xlim(0, 0.65)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_title('All 30 experimental conditions, sorted ascending by test macro F1',
                 fontsize=13, pad=12)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'figure_3_sorted_conditions.png',
                dpi=180, bbox_inches='tight')
    plt.close()


# =========================================================================
# FIGURE 4: grouped-by-model horizontal bar chart
# =========================================================================
def figure_4_grouped_by_model():
    """Horizontal grouped bars per FM: LP/FT x 1.0x/0.3x with distinct hatching."""
    method_style = {
        ('LP', '1.0x'):  {'alpha': 0.55, 'hatch': ''},
        ('LP', '0.3x'):  {'alpha': 0.55, 'hatch': '\\\\\\'},
        ('FT', '1.0x'):  {'alpha': 1.00, 'hatch': ''},
        ('FT', '0.3x'):  {'alpha': 1.00, 'hatch': '///'},
        ('Sup', '1.0x'): {'alpha': 1.00, 'hatch': ''},
        ('Sup', '0.3x'): {'alpha': 1.00, 'hatch': '///'},
    }

    methods_order_full = [('LP', '1.0x'), ('LP', '0.3x'),
                          ('FT', '1.0x'), ('FT', '0.3x')]
    methods_alphaearth = [('LP', '1.0x'), ('LP', '0.3x')]
    methods_supervised = [('Sup', '1.0x'), ('Sup', '0.3x')]
    methods_random = [('LP', '1.0x'), ('LP', '0.3x')]

    fig, ax = plt.subplots(figsize=(15, 12))
    bar_h = 0.19
    y_positions = []
    y_labels = []
    y_current = 0.0

    for fm in FM_ORDER:
        color = FM_COLORS[fm]
        if fm == 'AlphaEarth Foundations':
            methods = methods_alphaearth
        elif fm == 'Supervised ResNet-18':
            methods = methods_supervised
        elif fm == 'Random Features':
            methods = methods_random
        else:
            methods = methods_order_full

        fm_y_start = y_current
        for meth, frac in methods:
            d = load_agg(fm, meth, frac)
            if d is None:
                continue
            v, s = d['test_macro_f1']['mean'], d['test_macro_f1']['std']
            style = method_style[(meth, frac)]
            ax.barh(y_current, v, height=bar_h, color=color,
                    edgecolor='black', linewidth=0.6,
                    hatch=style['hatch'], alpha=style['alpha'],
                    xerr=s if s > 0 else None,
                    error_kw={'ecolor': 'black', 'capsize': 3, 'linewidth': 0.8})
            ax.text(v + max(s, 0.005) + 0.008, y_current, f'{v:.3f}',
                    va='center', fontsize=10)
            y_current += bar_h

        y_positions.append((fm_y_start + y_current - bar_h) / 2)
        y_labels.append(fm)
        y_current += bar_h * 0.8

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=13)
    ax.set_xlabel('Test macro F1', fontsize=14)
    ax.set_xlim(0, 0.65)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title('Test macro F1 grouped by model', fontsize=15, pad=15)

    style_legend = [
        Patch(facecolor='#cccccc', edgecolor='black', linewidth=0.6,
              alpha=0.55, label='LP 1.0x'),
        Patch(facecolor='#cccccc', edgecolor='black', linewidth=0.6,
              alpha=0.55, hatch='\\\\\\', label='LP 0.3x'),
        Patch(facecolor='#cccccc', edgecolor='black', linewidth=0.6,
              alpha=1.0, label='FT / Supervised 1.0x'),
        Patch(facecolor='#cccccc', edgecolor='black', linewidth=0.6,
              alpha=1.0, hatch='///', label='FT / Supervised 0.3x'),
    ]
    ax.legend(handles=style_legend, loc='lower right', fontsize=11,
              framealpha=0.95, title='Adaptation x Data scale',
              title_fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'figure_4_grouped_by_model.png',
                dpi=180, bbox_inches='tight')
    plt.close()


# =========================================================================
# FIGURE 5: FT-only labels-efficiency overlay
# =========================================================================
def figure_5_labels_efficiency_ft():
    """Fine-tuning labels-efficiency: 1.0x solid vs 0.3x dashed."""
    # FT-supporting FMs + Supervised ResNet-18
    ft_models = ['SatlasPretrain S1', 'SatlasPretrain S2', 'CROMA',
                 'Prithvi-EO-2.0', 'OlmoEarth v1.1-Base', 'DINOv3 ViT-L/16',
                 'Supervised ResNet-18']

    fig, ax = plt.subplots(figsize=(11, 7))
    for fm in ft_models:
        color = FM_COLORS[fm]
        is_baseline = fm == 'Supervised ResNet-18'
        lw = 1.5 if is_baseline else 2.0
        alpha = 0.75 if is_baseline else 1.0

        subdirs = RUN_DIR_MAP.get(fm, (None, None, None, None))
        ft_1x = subdirs[2]
        ft_3x = subdirs[3]

        # 1.0x solid
        curves = load_val_curves(ft_1x)
        if curves is not None:
            mean_c = curves.mean(axis=0)
            epochs = np.arange(1, len(mean_c) + 1)
            label = f'{fm} (baseline)' if is_baseline else fm
            ax.plot(epochs, mean_c, color=color, linestyle='-',
                    linewidth=lw, alpha=alpha, label=label)

        # 0.3x dashed
        curves = load_val_curves(ft_3x)
        if curves is not None:
            mean_c = curves.mean(axis=0)
            epochs = np.arange(1, len(mean_c) + 1)
            ax.plot(epochs, mean_c, color=color, linestyle='--',
                    linewidth=lw, alpha=alpha)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation macro F1', fontsize=13)
    ax.set_title('Fine-tuning labels-efficiency: 1.0x (solid) vs 0.3x (dashed) training data',
                 fontsize=14, pad=12)
    ax.set_xlim(0.5, 25.5)
    ax.set_ylim(0, 0.55)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'figure_5_labels_efficiency_ft.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# FIGURE 6 & 7: training dynamics panels with baselines
# =========================================================================
def _training_dynamics_panels(scale: str, out_name: str, super_title: str):
    """Two-panel LP + FT training dynamics figure at a given data scale.

    scale: '1.0x' or '0.3x'
    """
    # Which subdir index corresponds to (mode, scale)
    subdir_lookup = {
        ('LP', '1.0x'): 0,
        ('LP', '0.3x'): 1,
        ('FT', '1.0x'): 2,
        ('FT', '0.3x'): 3,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for panel_idx, mode in enumerate(['LP', 'FT']):
        ax = axes[panel_idx]
        idx = subdir_lookup[(mode, scale)]

        for fm in FM_ORDER:
            subdirs = RUN_DIR_MAP.get(fm, (None, None, None, None))
            subdir = subdirs[idx]
            curves = load_val_curves(subdir)
            if curves is None:
                continue

            mean_c = curves.mean(axis=0)
            std_c = curves.std(axis=0)
            epochs = np.arange(1, len(mean_c) + 1)

            color = FM_COLORS[fm]
            is_baseline = fm in ('Supervised ResNet-18', 'Random Features')
            linestyle = '--' if is_baseline else '-'
            linewidth = 1.8 if is_baseline else 2.0
            alpha_band = 0.10 if is_baseline else 0.15
            label = f'{fm} (baseline)' if is_baseline else fm

            ax.plot(epochs, mean_c, color=color, linestyle=linestyle,
                    linewidth=linewidth, label=label)
            ax.fill_between(epochs, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=alpha_band)

            # Star at mean best epoch
            be_mean = load_best_epoch_mean(subdir)
            if be_mean is not None and 1 <= be_mean <= len(mean_c):
                bidx = int(round(be_mean)) - 1
                if 0 <= bidx < len(mean_c):
                    ax.plot([be_mean], [mean_c[bidx]], marker='*',
                            markersize=14 if not is_baseline else 12,
                            color=color, markeredgecolor='black',
                            markeredgewidth=0.5, zorder=5)

        panel_label = {
            ('LP', '1.0x'): 'Linear Probing (LP), 1.0x training data',
            ('LP', '0.3x'): 'Linear Probing (LP), 0.3x training data',
            ('FT', '1.0x'): 'Fine-Tuning (FT) / Supervised, 1.0x training data',
            ('FT', '0.3x'): 'Fine-Tuning (FT) / Supervised, 0.3x training data',
        }[(mode, scale)]

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation macro F1', fontsize=12)
        ax.set_title(panel_label, fontsize=13)
        ax.set_xlim(0.5, 25.5)
        ax.set_ylim(0, 0.55)
        ax.grid(alpha=0.3, linestyle='--')

    # Combined legend
    handles, labels = [], []
    for ax in axes:
        h, ll = ax.get_legend_handles_labels()
        for hi, li in zip(h, ll):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=10, frameon=False)
    fig.suptitle(super_title, fontsize=15, y=1.00)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / out_name, dpi=150, bbox_inches='tight')
    plt.close()


def figure_6_training_dynamics_03x():
    """LP + FT panels at 0.3x training data, with baselines."""
    _training_dynamics_panels(
        scale='0.3x',
        out_name='figure_6_training_dynamics_0.3x.png',
        super_title='Training dynamics at 0.3x training data',
    )


def figure_7_training_dynamics_10x():
    """LP + FT panels at 1.0x training data, with baselines."""
    _training_dynamics_panels(
        scale='1.0x',
        out_name='figure_7_training_dynamics_1.0x.png',
        super_title='Training dynamics at 1.0x training data',
    )


# =========================================================================
# APPENDIX A: aggregate confusion matrices per model (Oranges palette)
# =========================================================================
def appendix_a_confusion_matrices():
    """9 aggregate confusion matrices, uniform Oranges styling.

    Each matrix uses the best-adaptation configuration per FM:
    - FT 1.0x for FT-supporting FMs
    - LP 1.0x for AlphaEarth (LP-only) and Random Features
    - Supervised 1.0x for ResNet-18
    """
    configs = [
        ('A1', 'SatlasPretrain S1 (FT, 1.0x)',            'fm_eval_satlas_s1_finetune_v1'),
        ('A2', 'SatlasPretrain S2 (FT, 1.0x)',            'fm_eval_satlas_s2_finetune_v1'),
        ('A3', 'CROMA (FT, 1.0x)',                        'fm_eval_croma_finetune_v1'),
        ('A4', 'Prithvi-EO-2.0 (FT, 1.0x)',               'fm_eval_prithvi_finetune_v1'),
        ('A5', 'AlphaEarth Foundations (LP, 1.0x)',       'fm_eval_alphaearth_v2_spatial'),
        ('A6', 'OlmoEarth v1.1-Base (FT, 1.0x)',          'fm_eval_olmoearth_finetune_v1'),
        ('A7', 'DINOv3 ViT-L/16 (FT, 1.0x)',              'fm_eval_dinov3_finetune_v1'),
        ('A8', 'Supervised ResNet-18 (1.0x)',             'fm_eval_resnet18_supervised_v1'),
        ('A9', 'Random Features (LP, 1.0x)',              'fm_eval_resnet18_random_features_v1'),
    ]

    for fig_label, label, subdir in configs:
        cm = load_summed_confusion_matrix(subdir)
        if cm is None:
            print(f"  ! {fig_label} - no data at {subdir}")
            continue

        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums_safe = np.where(row_sums == 0, 1, row_sums)
        cm_norm = cm / row_sums_safe

        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(cm_norm, cmap='Oranges', vmin=0, vmax=1)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                raw = int(cm[i, j])
                pct = cm_norm[i, j] * 100
                color = 'white' if cm_norm[i, j] > 0.55 else 'black'
                if pct >= 0.5:
                    txt = f'{raw}\n({pct:.0f}%)'
                elif raw > 0:
                    txt = f'{raw}'
                else:
                    txt = ''
                ax.text(j, i, txt, ha='center', va='center',
                        color=color, fontsize=11)

        ax.set_xticks(range(len(CLASS_LABELS)))
        ax.set_yticks(range(len(CLASS_LABELS)))
        ax.set_xticklabels(CLASS_LABELS, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(CLASS_LABELS, fontsize=12)
        ax.set_xlabel('Predicted class', fontsize=14, labelpad=10)
        ax.set_ylabel('True class', fontsize=14, labelpad=10)
        ax.set_title(f'Figure {fig_label}. {label} - aggregate confusion matrix '
                     f'(3 seeds summed, n = {cm.sum()})',
                     fontsize=14, pad=18)

        cbar = plt.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label('Row-normalized prediction rate', fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        plt.tight_layout()
        slug = label.split(' (')[0].replace(' ', '_').replace('/', '-').lower()
        plt.savefig(OUTPUTS_DIR / f'{fig_label}_cm_{slug}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()


# =========================================================================
# APPENDIX E: per-sector and per-region multi-panel charts
# =========================================================================
def _best_condition_agg(fm: str) -> Optional[dict]:
    """Return the aggregate JSON for each FM's best-adaptation condition."""
    best = {
        'SatlasPretrain S1':      ('FT', '1.0x'),
        'SatlasPretrain S2':      ('FT', '1.0x'),
        'CROMA':                  ('FT', '1.0x'),
        'Prithvi-EO-2.0':         ('FT', '1.0x'),
        'AlphaEarth Foundations': ('LP', '1.0x'),
        'OlmoEarth v1.1-Base':    ('FT', '1.0x'),
        'DINOv3 ViT-L/16':        ('FT', '1.0x'),
        'Supervised ResNet-18':   ('Sup', '1.0x'),
        'Random Features':        ('LP', '1.0x'),
    }
    mode, scale = best[fm]
    return load_agg(fm, mode, scale)


def appendix_e1_per_sector():
    """4-panel per-sector chart with sample counts and class counts."""
    sectors = [
        ('transport', 'Transport',  865, 3),
        ('telecom',   'Telecom',    100, 1),
        ('water',     'Water',      920, 3),
        ('energy',    'Energy',     928, 6),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = axes.flatten()

    for i, (sec_key, sec_name, n, n_classes) in enumerate(sectors):
        ax = axes_flat[i]
        vals, stds, colors, labels = [], [], [], []
        for fm in FM_ORDER:
            d = _best_condition_agg(fm)
            if d is None:
                continue
            entry = d['per_sector_f1'][sec_key]
            vals.append(entry['mean_macro_f1'])
            stds.append(entry['std_macro_f1'])
            colors.append(FM_COLORS[fm])
            labels.append(fm)

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, xerr=stds, color=colors, edgecolor='black',
                linewidth=0.5,
                error_kw={'ecolor': 'black', 'capsize': 3, 'linewidth': 0.8})
        for j, (v, s) in enumerate(zip(vals, stds)):
            ax.text(v + max(s, 0.005) + 0.008, j, f'{v:.3f}',
                    va='center', fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlim(0, 0.95)
        ax.set_xlabel('Test Macro F1', fontsize=11)
        cls_word = 'class' if n_classes == 1 else 'classes'
        ax.set_title(f'{sec_name} (n = {n}, {n_classes} {cls_word})', fontsize=12)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()

    fig.suptitle('Per-Sector Test Macro F1 by Foundation Model (best-condition per FM)',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'appendix_E1_per_sector.png',
                dpi=180, bbox_inches='tight')
    plt.close()


def appendix_e2_per_region():
    """7-panel per-region chart with sample counts (one empty subplot)."""
    regions = [
        ('north-america',     'North America',      402),
        ('south-america',     'South America',      429),
        ('central-america',   'Central America',    357),
        ('europe',            'Europe',             303),
        ('africa',            'Africa',             434),
        ('asia',              'Asia',               433),
        ('australia-oceania', 'Australia / Oceania', 455),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes_flat = axes.flatten()

    for i, (reg_key, reg_name, n) in enumerate(regions):
        ax = axes_flat[i]
        vals, stds, colors, labels = [], [], [], []
        for fm in FM_ORDER:
            d = _best_condition_agg(fm)
            if d is None:
                continue
            entry = d['per_region_f1'][reg_key]
            vals.append(entry['mean_macro_f1'])
            stds.append(entry['std_macro_f1'])
            colors.append(FM_COLORS[fm])
            labels.append(fm)

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, xerr=stds, color=colors, edgecolor='black',
                linewidth=0.5,
                error_kw={'ecolor': 'black', 'capsize': 3, 'linewidth': 0.8})
        for j, (v, s) in enumerate(zip(vals, stds)):
            ax.text(v + max(s, 0.005) + 0.008, j, f'{v:.3f}',
                    va='center', fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlim(0, 0.75)
        ax.set_xlabel('Test Macro F1', fontsize=11)
        ax.set_title(f'{reg_name} (n = {n})', fontsize=12)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()

    axes_flat[7].axis('off')

    fig.suptitle('Per-Region Test Macro F1 by Foundation Model (best-condition per FM)',
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'appendix_E2_per_region.png',
                dpi=180, bbox_inches='tight')
    plt.close()


# =========================================================================
# MAIN
# =========================================================================
FIGURE_REGISTRY = {
    '3': figure_3_sorted_conditions,
    '4': figure_4_grouped_by_model,
    '5': figure_5_labels_efficiency_ft,
    '6': figure_6_training_dynamics_03x,
    '7': figure_7_training_dynamics_10x,
}

APPENDIX_REGISTRY = {
    'A': appendix_a_confusion_matrices,
    'E1': appendix_e1_per_sector,
    'E2': appendix_e2_per_region,
}


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate Infra-Bench CLS paper figures.'
    )
    parser.add_argument('--figure', type=str, default=None,
                        help='Generate only this figure (3-7)')
    parser.add_argument('--appendix', type=str, default=None,
                        help='Generate only this appendix figure set (A, E1, E2)')
    parser.add_argument('--all', action='store_true',
                        help='Generate everything (default if no other flag)')
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.figure is not None:
        if args.figure not in FIGURE_REGISTRY:
            raise ValueError(f'Unknown figure: {args.figure}. '
                             f'Options: {list(FIGURE_REGISTRY.keys())}')
        print(f'Generating figure {args.figure}...')
        FIGURE_REGISTRY[args.figure]()
    elif args.appendix is not None:
        if args.appendix not in APPENDIX_REGISTRY:
            raise ValueError(f'Unknown appendix: {args.appendix}. '
                             f'Options: {list(APPENDIX_REGISTRY.keys())}')
        print(f'Generating appendix {args.appendix}...')
        APPENDIX_REGISTRY[args.appendix]()
    else:
        # Default: generate everything
        for name, fn in FIGURE_REGISTRY.items():
            print(f'Generating figure {name}...')
            fn()
        for name, fn in APPENDIX_REGISTRY.items():
            print(f'Generating appendix {name}...')
            fn()

    print(f'Done. Figures written to {OUTPUTS_DIR}/')


if __name__ == '__main__':
    main()
