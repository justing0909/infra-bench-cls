"""Build the canonical 10-class results.json for the Infra-Bench CLS site.

Ports the 10-class re-fit (paper_figures.ipynb cell 6) and the Appendix F
CM-derived metrics (cell 25) so every number on the site matches the paper.
"""
import glob
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np

RESULTS_BASE = Path(sys.argv[1])
OUT = Path(sys.argv[2])

# ---------------------------------------------------------------- taxonomy
CLASS_COUNTS_13 = [105, 141, 511, 70, 98, 3, 228, 118, 574, 125, 737, 3, 100]
EXCLUDED_IDX = {5, 7, 11}          # wind_farm, water_works, port_terminal
KEEP_IDX = [i for i in range(13) if i not in EXCLUDED_IDX]
DENOM_10 = sum(CLASS_COUNTS_13[i] for i in KEEP_IDX)   # 2689

CLASS_DISPLAY_ALL_13 = [
    'Transmission Substation', 'Distribution Substation', 'Distribution (Other)',
    'Power Plant', 'Solar Farm', 'Wind Farm',
    'Wastewater Plant', 'Water Works', 'Storage Tank',
    'Airport', 'Train Station', 'Port Terminal',
    'Data Center',
]
CLASS_KEYS_13 = [
    'energy.transmission.substation', 'energy.distribution.substation',
    'energy.distribution.other', 'energy.generation.power_plant',
    'energy.generation.solar_farm', 'energy.generation.wind_farm',
    'water.wastewater_plant', 'water.water_works', 'water.storage_tank',
    'transport.airport', 'transport.train_station', 'transport.port_terminal',
    'telecom.data_center',
]
SECTOR_CLASSES_10 = {
    'energy':    [0, 1, 2, 3, 4],
    'water':     [6, 8],
    'transport': [9, 10],
    'telecom':   [12],
}
REGIONS = ['north-america', 'south-america', 'central-america', 'europe',
           'africa', 'asia', 'australia-oceania']
REGION_DISPLAY = {
    'north-america': 'North America', 'south-america': 'South America',
    'central-america': 'Central America', 'europe': 'Europe',
    'africa': 'Africa', 'asia': 'Asia',
    'australia-oceania': 'Australia / Oceania',
}

# ---------------------------------------------------------------- run layout
FM_ORDER = [
    'SatlasPretrain S1', 'SatlasPretrain S2', 'CROMA', 'Prithvi-EO-2.0',
    'AlphaEarth Foundations', 'OlmoEarth v1.1-Base', 'DINOv3 ViT-L/16',
    'Supervised ResNet-18', 'Random Features',
]
FM_COLORS = {
    'SatlasPretrain S1': '#e67e22', 'SatlasPretrain S2': '#1f77b4',
    'CROMA': '#e6b800', 'Prithvi-EO-2.0': '#9467bd',
    'AlphaEarth Foundations': '#2ca02c', 'OlmoEarth v1.1-Base': '#17becf',
    'DINOv3 ViT-L/16': '#d62728', 'Supervised ResNet-18': '#b0b0b0',
    'Random Features': '#8B4513',
}
IS_BASELINE = {'Supervised ResNet-18', 'Random Features'}

RUN_DIR_MAP = {
    'SatlasPretrain S1':      ('fm_eval_satlas_s1_v2_spatial', 'fm_eval_satlas_s1_lp_0.3x_v1',
                               'fm_eval_satlas_s1_finetune_v1', 'fm_eval_satlas_s1_finetune_0.3x_v1'),
    'SatlasPretrain S2':      ('fm_eval_satlaspretrain_v2_spatial', 'fm_eval_satlas_s2_lp_0.3x_v1',
                               'fm_eval_satlas_s2_finetune_v1', 'fm_eval_satlas_s2_finetune_0.3x_v1'),
    'CROMA':                  ('fm_eval_croma_v2_spatial', 'fm_eval_croma_lp_0.3x_v1',
                               'fm_eval_croma_finetune_v1', 'fm_eval_croma_finetune_0.3x_v1'),
    'Prithvi-EO-2.0':         ('fm_eval_prithvi_v2_spatial', 'fm_eval_prithvi_lp_0.3x_v1',
                               'fm_eval_prithvi_finetune_v1', 'fm_eval_prithvi_finetune_0.3x_v1'),
    'AlphaEarth Foundations': ('fm_eval_alphaearth_v2_spatial', 'fm_eval_alphaearth_lp_0.3x_v1',
                               None, None),
    'OlmoEarth v1.1-Base':    ('fm_eval_olmoearth_lp_v1', 'fm_eval_olmoearth_lp_0.3x_v1',
                               'fm_eval_olmoearth_finetune_v1', 'fm_eval_olmoearth_finetune_0.3x_v1'),
    'DINOv3 ViT-L/16':        ('fm_eval_dinov3_lp_v1', 'fm_eval_dinov3_lp_0.3x_v1',
                               'fm_eval_dinov3_finetune_v1', 'fm_eval_dinov3_finetune_0.3x_v1'),
    'Supervised ResNet-18':   (None, None,
                               'fm_eval_resnet18_supervised_v1', 'fm_eval_resnet18_supervised_0.3x_v1'),
    'Random Features':        ('fm_eval_resnet18_random_features_v1', 'fm_eval_resnet18_random_features_0.3x_v1',
                               None, None),
}
MODE_SCALE_TO_IDX = {
    ('LP', '1.0x'): 0, ('LP', '0.3x'): 1,
    ('FT', '1.0x'): 2, ('FT', '0.3x'): 3,
    ('Sup', '1.0x'): 2, ('Sup', '0.3x'): 3,
}
WRAPPER = {'LP': 'linear_probe', 'FT': 'finetune', 'Sup': 'linear_probe'}
SEEDS = (314, 271, 161)

# GMACs / params from evaluation/analysis/find_GMACs.ipynb.
# `proxy` marks a timm architecture stand-in used where the official loader
# was unavailable; GMACs is a function of architecture, not trained weights.
COST = {
    'SatlasPretrain S1':      {'gmacs': 247.37, 'params_m': 86.7,  'proxy': True},
    'SatlasPretrain S2':      {'gmacs': 248.09, 'params_m': 86.8,  'proxy': True},
    'CROMA':                  {'gmacs': 703.16, 'params_m': 194.4, 'proxy': False},
    'Prithvi-EO-2.0':         {'gmacs': 957.60, 'params_m': 304.1, 'proxy': True},
    'AlphaEarth Foundations': {'gmacs': 0.00,   'params_m': 0.0,   'proxy': False},
    'OlmoEarth v1.1-Base':    {'gmacs': 275.42, 'params_m': 87.6,  'proxy': True},
    'DINOv3 ViT-L/16':        {'gmacs': 955.13, 'params_m': 303.3, 'proxy': True},
    'Supervised ResNet-18':   {'gmacs': 32.87,  'params_m': 11.2,  'proxy': False},
    'Random Features':        {'gmacs': 32.87,  'params_m': 11.2,  'proxy': False},
}


def pop_std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def stat(vals):
    return {'mean': mean(vals), 'std': pop_std(vals), 'per_seed': list(vals)}


def conditions():
    """Yield (fm, mode, scale) for all 30 real conditions."""
    for fm in FM_ORDER:
        if fm == 'Supervised ResNet-18':
            modes = ('Sup',)
        elif fm in ('AlphaEarth Foundations', 'Random Features'):
            modes = ('LP',)
        else:
            modes = ('LP', 'FT')
        for mode in modes:
            for scale in ('1.0x', '0.3x'):
                yield fm, mode, scale


def run_dir(fm, mode, scale):
    idx = MODE_SCALE_TO_IDX.get((mode, scale))
    subdirs = RUN_DIR_MAP.get(fm)
    if idx is None or subdirs is None or subdirs[idx] is None:
        return None
    return RESULTS_BASE / subdirs[idx]


def load_agg(fm, mode, scale):
    d = run_dir(fm, mode, scale)
    if d is None:
        return None
    hits = sorted(glob.glob(str(d / '*_aggregate.json')))
    if not hits:
        return None
    with open(hits[0], encoding='utf-8') as f:
        return json.load(f)


def load_cms(fm, mode, scale):
    """Per-seed 13x13 confusion matrices, keyed by seed."""
    d = run_dir(fm, mode, scale)
    if d is None:
        return {}
    out = {}
    for seed in SEEDS:
        hits = sorted(glob.glob(str(d / f'*_seed{seed}_results.json')))
        hits = [h for h in hits if 'smoke' not in os.path.basename(h).lower()]
        if not hits:
            continue
        with open(hits[0], encoding='utf-8') as f:
            r = json.load(f)
        out[seed] = np.array(r[WRAPPER[mode]]['test']['confusion'], dtype=np.int64)
    return out


def cm_metrics_10(cm_13):
    """10-class accuracy / macro P / macro R / weighted P from one 13x13 CM.

    Recall uses original 13-col row sums (= class supports) so per-class recall
    matches the paper's Tables D1/D2. Precision uses sliced column sums.
    """
    cm_10 = cm_13[np.ix_(KEEP_IDX, KEEP_IDX)]
    tp = np.diag(cm_10).astype(np.float64)
    row_orig = cm_13[KEEP_IDX].sum(axis=1).astype(np.float64)
    col_10 = cm_10.sum(axis=0).astype(np.float64)

    recall = np.divide(tp, row_orig, out=np.zeros_like(tp), where=row_orig > 0)
    precision = np.divide(tp, col_10, out=np.zeros_like(tp), where=col_10 > 0)
    total = float(row_orig.sum())
    return {
        'accuracy': float(tp.sum() / total),
        'macro_precision': float(precision.mean()),
        'macro_recall': float(recall.mean()),
        'weighted_precision': float((row_orig * precision).sum() / total),
    }


def build_condition(fm, mode, scale):
    agg = load_agg(fm, mode, scale)
    if agg is None:
        return None
    pcf = agg.get('per_class_f1')
    if not pcf or len(pcf) != 13:
        raise SystemExit(f'unexpected per_class_f1 schema for {fm}/{mode}/{scale}')
    n_seeds = len(pcf[0]['per_seed'])

    # --- 10-class macro / weighted F1, recomputed per seed then aggregated
    macro, weighted = [], []
    for s in range(n_seeds):
        f1s = [pcf[i]['per_seed'][s] for i in range(13)]
        macro.append(mean(f1s[i] for i in KEEP_IDX))
        weighted.append(
            sum(CLASS_COUNTS_13[i] * f1s[i] for i in KEEP_IDX) / DENOM_10
        )

    # --- per-class F1 (invariant to which other classes are aggregated over)
    per_class = [{
        'key': CLASS_KEYS_13[i],
        'name': CLASS_DISPLAY_ALL_13[i],
        'n': CLASS_COUNTS_13[i],
        'mean': pcf[i]['mean_f1'],
        'std': pcf[i]['std_f1'],
        'per_seed': pcf[i]['per_seed'],
    } for i in KEEP_IDX]

    # --- per-sector F1 on retained classes only
    per_sector = {}
    for sect, idxs in SECTOR_CLASSES_10.items():
        vals = [mean(pcf[i]['per_seed'][s] for i in idxs) for s in range(n_seeds)]
        per_sector[sect] = dict(
            stat(vals), n=sum(CLASS_COUNTS_13[i] for i in idxs)
        )

    # --- per-region: 13-class ONLY. The aggregates and the per-seed files both
    # store per-region macro F1 already reduced over classes, so there is no way
    # to re-fit this to 10 classes. Flagged for the UI.
    per_region = {}
    for reg in REGIONS:
        e = (agg.get('per_region_f1') or {}).get(reg)
        if e:
            per_region[reg] = {
                'mean': e['mean_macro_f1'], 'std': e['std_macro_f1'],
                'per_seed': e['per_seed'], 'n': e['n'], 'class_basis': 13,
            }

    # --- CM-derived metrics (Appendix F)
    cms = load_cms(fm, mode, scale)
    cm_derived, cm_sum = {}, None
    if cms:
        per_seed_m = {s: cm_metrics_10(cm) for s, cm in cms.items()}
        for k in ('accuracy', 'macro_precision', 'macro_recall', 'weighted_precision'):
            cm_derived[k] = stat([per_seed_m[s][k] for s in cms])
        stacked = np.sum([cms[s] for s in cms], axis=0)
        cm_sum = stacked[np.ix_(KEEP_IDX, KEEP_IDX)].tolist()

    rec = {
        'fm': fm, 'protocol': mode, 'scale': scale,
        'is_baseline': fm in IS_BASELINE,
        'seeds': agg.get('seeds', list(SEEDS)),
        'macro_f1': stat(macro),
        'weighted_f1': stat(weighted),
        'per_class_f1': per_class,
        'per_sector_f1': per_sector,
        'per_region_f1': per_region,
        'confusion_10': cm_sum,
    }
    rec.update(cm_derived)

    for key in ('peak_gpu_gb', 'wall_time_s'):
        if key in agg and isinstance(agg[key], dict) and 'mean' in agg[key]:
            rec[key] = agg[key]['mean']
    return rec


def main():
    rows = []
    for fm, mode, scale in conditions():
        rec = build_condition(fm, mode, scale)
        if rec is None:
            print(f'  MISSING {fm} {mode} {scale}')
            continue
        rows.append(rec)

    out = {
        'meta': {
            'benchmark': 'Infra-Bench CLS',
            'class_basis': 10,
            'excluded_classes': [
                {'idx': 5,  'key': CLASS_KEYS_13[5],  'name': 'Wind Farm',
                 'n': 3,   'reason': 'low test-n'},
                {'idx': 11, 'key': CLASS_KEYS_13[11], 'name': 'Port Terminal',
                 'n': 3,   'reason': 'low test-n'},
                {'idx': 7,  'key': CLASS_KEYS_13[7],  'name': 'Water Works',
                 'n': 118, 'reason': 'class ambiguity'},
            ],
            'test_n_per_seed': DENOM_10,
            'test_n_per_seed_13class': sum(CLASS_COUNTS_13),
            'seeds': list(SEEDS),
            'fm_order': FM_ORDER,
            'fm_colors': FM_COLORS,
            'region_display': REGION_DISPLAY,
            'class_names_10': [CLASS_DISPLAY_ALL_13[i] for i in KEEP_IDX],
            'class_n_10': [CLASS_COUNTS_13[i] for i in KEEP_IDX],
            'cost': COST,
            'per_region_class_basis': 13,
            'per_region_note': (
                'Per-region macro F1 is computed on the full 13-class taxonomy. '
                'The source results store per-region F1 already reduced over '
                'classes, so it cannot be re-fit to the 10-class subset.'
            ),
        },
        'conditions': rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(rows)} conditions)')


if __name__ == '__main__':
    main()
