import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / 'REPORT.json').read_text())
records = list((root / 'records').glob('*.json'))
assert len(records) == 45
assert len(list((root / 'diagnostics/records').glob('*.json'))) == 9
checked = 0
for line in (root / 'SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ', 1)
    path = root / name
    if path.suffix == '.npz':
        continue
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    checked += 1
print('LOCAL_METADATA_HASHES_PASS', checked)
rates = {}
for row in report['timing']:
    fw, w = row['framework'], row['world_size']
    paths = [root / 'records' / f'{fw}-w{w}-r{r}.json' for r in range(1, 6)]
    data = [json.loads(p.read_text()) for p in paths]
    for d in data:
        assert (d['global_batch'], d['updates'], d['n_eff']) == (256, 390, 99840)
        rank_times = d.get('rank_seconds') or [r['seconds'] for r in d['ranks']]
        assert d['seconds'] == max(rank_times)
    times = [d['seconds'] for d in data]
    median = statistics.median(times)
    assert median == row['median_s']
    rate = 99840 / median
    assert math.isclose(rate, row['images_per_second'], rel_tol=1e-14)
    rates[fw, w] = rate
    speed = rate / rates[fw, 1]
    assert math.isclose(speed, row['speedup'], rel_tol=1e-14)
    assert math.isclose(speed/w, row['efficiency'], rel_tol=1e-14)
    print(fw, w, 'times', times, 'median', median, 'img/s', rate, 'speedup', speed)
for a,b,w in [('flux-historical','flux-current',4),('flux-current','flux-historical',1),('flux-current','pytorch',1),('flux-current','pytorch',2),('flux-current','pytorch',4)]:
    print('THROUGHPUT_PERCENT',a,b,w,100*(rates[a,w]/rates[b,w]-1))
comparisons = report['diagnostics']['comparisons']
assert len(comparisons) == 15
for category in ('param','m','v','beta_power'):
    rows = [c for comparison in comparisons for c in comparison['categories'] if c['category']==category]
    print('DIAGNOSTIC',category,'rel_l2_range',min(c['rel_l2'] for c in rows),max(c['rel_l2'] for c in rows),'max_abs',max(c['max_abs'] for c in rows),'exact_arrays',sorted(set(c['exact_arrays'] for c in rows)))
print('CLAIM_ARITHMETIC_PASS')
