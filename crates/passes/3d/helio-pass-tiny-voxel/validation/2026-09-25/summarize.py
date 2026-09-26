import csv
import json
import math
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parent
results = {}
for name in sys.argv[1:]:
    directory = root / name
    rows = list(csv.DictReader((directory / 'frames.csv').open()))
    stages = {}
    for stage in ('walk', '200m', '1km', 'orbit', 'descent', 'resize', 'orbital_edit', '1m_base'):
        selected = [r for r in rows if r['stage'] == stage]
        if stage in ('200m', '1km', 'orbit'):
            selected = selected[-60:]
        elif stage in ('orbital_edit', '1m_base'):
            selected = selected[-30:]
        times = sorted(float(r['sync_frame_ms']) for r in selected)
        stages[stage] = {
            'frames': len(times),
            'p50_ms': times[math.ceil(len(times) * .50) - 1],
            'p95_ms': times[math.ceil(len(times) * .95) - 1],
            'max_ms': max(times),
            'refining_frames': sum(r['refining'] == 'true' for r in selected),
        }
    log = (root / (name + '.log')).read_text(encoding='utf-8-sig', errors='replace')
    hits = []
    for path in directory.glob('*.hits.csv'):
        if path.stem.startswith('composition-sentinel'):
            continue
        hit = next(csv.DictReader(path.open()))
        hits.append({**{k: int(v) for k, v in hit.items()}, 'file': path.name})
    result = {
        'frames': len(rows),
        'complete': 'VOXEL_FLIGHT_COMPLETE' in log,
        'config': re.findall(r'VOXEL_FLIGHT_CONFIG[^\r\n]+', log),
        'adapter': re.findall(r'VOXEL_FLIGHT_ADAPTER[^\r\n]+', log),
        'stages': stages,
        'settlement_ms': {m[0]: float(m[1]) for m in re.findall(r'VOXEL_FLIGHT_SETTLED stage=(\S+) load_ms=([\d.]+)', log)},
        'not_ready_after_initial_load': sum(r['ready'] != 'true' and r['stage'] != 'ground_load' for r in rows),
        'capture_audits': len(hits),
        'exhausted_hits': sum(h['exhausted'] for h in hits),
        'loading_hits': sum(h['loading'] for h in hits),
        'minimum_solid_hits': min(h['solid'] for h in hits),
        'audits': hits,
    }
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    results[name] = {k:v for k,v in result.items() if k != 'audits'}
print(json.dumps(results, indent=2))
