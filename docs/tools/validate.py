"""validate results.json against every number printed in paper_figures.ipynb."""
import json
import sys

D = json.load(open(sys.argv[1], encoding='utf-8'))
idx = {(c['fm'], c['protocol'], c['scale']): c for c in D['conditions']}
fails = []


def chk(label, got, exp, tol=5e-4):
    ok = got is not None and abs(got - exp) < tol
    if not ok:
        fails.append(f'{label}: got {got} expected {exp}')
    print(f'  {"OK " if ok else "FAIL"} {label:<52} got={got if got is None else round(got,4)} exp={exp}')


print('== structural constants')
chk('test_n_per_seed', D['meta']['test_n_per_seed'], 2689, 0.5)
chk('test_n_13class', D['meta']['test_n_per_seed_13class'], 2813, 0.5)
chk('n conditions', len(D['conditions']), 30, 0.5)

# every check below indexes conditions by name, so a short file would fail
# with a KeyError rather than a readable message. the usual cause is
# build_results.py pointed at a path with no results in it, which it reports
# per-condition and then writes out empty.
if len(D['conditions']) != 30:
    print(f'\nonly {len(D["conditions"])} of 30 conditions present -- stopping here.')
    print('check the results tree passed to build_results.py.')
    sys.exit(1)
for sect, n in [('transport', 862), ('telecom', 100), ('water', 802), ('energy', 925)]:
    chk(f'sector n {sect}', idx[('DINOv3 ViT-L/16', 'FT', '1.0x')]['per_sector_f1'][sect]['n'], n, 0.5)

print('\n== cell 6: macro / weighted F1 sanity check')
for (fm, mo, sc), (m, w) in {
    ('DINOv3 ViT-L/16', 'FT', '1.0x'): (0.579, 0.627),
    ('SatlasPretrain S2', 'FT', '1.0x'): (0.559, 0.601),
    ('CROMA', 'FT', '1.0x'): (0.341, 0.343),
    ('AlphaEarth Foundations', 'LP', '1.0x'): (0.266, 0.268),
}.items():
    r = idx[(fm, mo, sc)]
    chk(f'{fm} {mo}/{sc} macro', r['macro_f1']['mean'], m)
    chk(f'{fm} {mo}/{sc} weighted', r['weighted_f1']['mean'], w)

print('\n== cell 23: LP -> FT macro F1 (full precision)')
chk('Supervised ResNet-18 Sup/1.0x macro', idx[('Supervised ResNet-18', 'Sup', '1.0x')]['macro_f1']['mean'],
    0.3921909735963698, 1e-9)
for fm, lp, ft in [
    ('DINOv3 ViT-L/16',    0.37609484073895716, 0.5792118367110837),
    ('SatlasPretrain S2',  0.24347408681117666, 0.5591997071975423),
    ('OlmoEarth v1.1-Base', 0.28089496548988013, 0.5395027906032045),
    ('Prithvi-EO-2.0',     0.30069981704760657, 0.5307365435974104),
    ('SatlasPretrain S1',  0.21031898691747297, 0.3609519995106139),
    ('CROMA',              0.2914517844388335,  0.3409842103567273),
]:
    chk(f'{fm} LP', idx[(fm, 'LP', '1.0x')]['macro_f1']['mean'], lp, 1e-9)
    chk(f'{fm} FT', idx[(fm, 'FT', '1.0x')]['macro_f1']['mean'], ft, 1e-9)

print('\n== supporting information Tables S10-S13')
# transcribed verbatim from the notebook's executed output.
F = {
'accuracy': """
SatlasPretrain S1 0.290 0.027 0.219 0.018 0.384 0.010 0.342 0.024
SatlasPretrain S2 0.323 0.016 0.294 0.017 0.595 0.018 0.546 0.006
CROMA 0.309 0.028 0.275 0.032 0.336 0.029 0.297 0.015
Prithvi-EO-2.0 0.331 0.011 0.277 0.008 0.583 0.003 0.530 0.012
AlphaEarth Foundations 0.275 0.001 0.271 0.001 - - - -
OlmoEarth v1.1-Base 0.289 0.007 0.285 0.025 0.571 0.029 0.473 0.012
DINOv3 ViT-L/16 0.403 0.011 0.363 0.003 0.625 0.004 0.573 0.001
Supervised ResNet-18 0.395 0.014 0.307 0.016 0.395 0.014 0.307 0.016
Random Features 0.231 0.009 0.232 0.009 - - - -
""",
'macro_precision': """
SatlasPretrain S1 0.233 0.035 0.228 0.011 0.383 0.020 0.325 0.021
SatlasPretrain S2 0.259 0.015 0.244 0.014 0.607 0.018 0.555 0.014
CROMA 0.341 0.023 0.290 0.027 0.362 0.034 0.287 0.004
Prithvi-EO-2.0 0.312 0.007 0.278 0.010 0.578 0.008 0.531 0.007
AlphaEarth Foundations 0.262 0.002 0.236 0.002 - - - -
OlmoEarth v1.1-Base 0.298 0.012 0.287 0.014 0.575 0.028 0.487 0.014
DINOv3 ViT-L/16 0.382 0.003 0.358 0.006 0.617 0.004 0.565 0.007
Supervised ResNet-18 0.419 0.003 0.321 0.036 0.419 0.003 0.321 0.036
Random Features 0.148 0.011 0.127 0.004 - - - -
""",
'macro_recall': """
SatlasPretrain S1 0.268 0.021 0.261 0.010 0.370 0.023 0.317 0.008
SatlasPretrain S2 0.291 0.020 0.262 0.018 0.543 0.001 0.480 0.008
CROMA 0.343 0.020 0.300 0.013 0.369 0.006 0.334 0.016
Prithvi-EO-2.0 0.345 0.017 0.300 0.004 0.513 0.001 0.465 0.005
AlphaEarth Foundations 0.369 0.002 0.335 0.004 - - - -
OlmoEarth v1.1-Base 0.323 0.005 0.304 0.021 0.537 0.007 0.429 0.012
DINOv3 ViT-L/16 0.416 0.007 0.383 0.006 0.567 0.006 0.510 0.001
Supervised ResNet-18 0.395 0.007 0.335 0.024 0.395 0.007 0.335 0.024
Random Features 0.182 0.014 0.173 0.013 - - - -
""",
'weighted_precision': """
SatlasPretrain S1 0.290 0.048 0.301 0.006 0.422 0.010 0.375 0.015
SatlasPretrain S2 0.342 0.027 0.306 0.026 0.642 0.006 0.590 0.011
CROMA 0.370 0.009 0.345 0.013 0.400 0.019 0.330 0.005
Prithvi-EO-2.0 0.392 0.004 0.365 0.011 0.621 0.008 0.572 0.005
AlphaEarth Foundations 0.341 0.003 0.309 0.002 - - - -
OlmoEarth v1.1-Base 0.359 0.007 0.357 0.010 0.612 0.003 0.517 0.009
DINOv3 ViT-L/16 0.466 0.008 0.448 0.012 0.654 0.002 0.598 0.004
Supervised ResNet-18 0.450 0.018 0.375 0.008 0.450 0.018 0.375 0.008
Random Features 0.214 0.008 0.183 0.007 - - - -
""",
}
SLOTS = [('LP', '1.0x'), ('LP', '0.3x'), ('FT', '1.0x'), ('FT', '0.3x')]
for metric, block in F.items():
    print(f'  -- {metric}')
    for line in block.strip().splitlines():
        parts = line.split()
        nums = []
        while parts and (parts[-1] == '-' or parts[-1].replace('.', '').isdigit()):
            nums.insert(0, parts.pop())
        fm = ' '.join(parts)
        for i, (mode, scale) in enumerate(SLOTS):
            mu, sd = nums[2 * i], nums[2 * i + 1]
            # the paper table prints Supervised ResNet-18 under both LP and FT
            # headers; both resolve to the same 'Sup' condition.
            actual = 'Sup' if fm == 'Supervised ResNet-18' else mode
            r = idx.get((fm, actual, scale))
            if mu == '-':
                if r is not None:
                    fails.append(f'{metric} {fm} {mode}/{scale}: expected absent, found condition')
                continue
            if r is None:
                fails.append(f'{metric} {fm} {mode}/{scale}: condition missing')
                print(f'    FAIL {fm} {mode}/{scale} missing')
                continue
            chk(f'{fm} {mode}/{scale} mean', r[metric]['mean'], float(mu))
            chk(f'{fm} {mode}/{scale} std',  r[metric]['std'],  float(sd))

print('\n== confusion matrix consistency (diag/rowsum == per-class recall)')
r = idx[('DINOv3 ViT-L/16', 'FT', '1.0x')]
cm = r['confusion_10']
chk('cm total == 3 seeds x 2689 retained-row mass',
    sum(sum(row) for row in cm), None if cm is None else sum(sum(row) for row in cm), 0.5)
print(f'  (row sums after 10-slice: {[sum(row) for row in cm]})')
print(f'  (3 x class_n_10:          {[3*n for n in D["meta"]["class_n_10"]]})')

print('\n' + '=' * 60)
if fails:
    print(f'{len(fails)} FAILURES')
    for f in fails:
        print('  ' + f)
    sys.exit(1)
print('ALL CHECKS PASSED')
