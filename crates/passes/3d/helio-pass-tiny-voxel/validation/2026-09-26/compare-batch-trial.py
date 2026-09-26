import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
results = {}
for resolution in ('720p', '1080p-quality'):
    names = [f'trial-{kind}-{resolution}' for kind in ('control', '1024')]
    summaries = [json.loads((root / name / 'summary.json').read_text()) for name in names]
    cuts = []
    for name in names:
        rows = list(csv.DictReader((root / name / 'frames.csv').open()))
        cuts.append({stage: {key: next(r for r in reversed(rows) if r['stage'] == stage)[key]
                            for key in ('bricks', 'pixel_budget')}
                     for stage in ('200m', '1km', 'orbit', 'returned_ground', 'resize_settle', 'orbital_edit', '1m_base')})
    control, candidate = summaries
    ratios = {stage: {metric: candidate['stages'][stage][metric] / control['stages'][stage][metric]
                      for metric in ('p95_ms', 'max_ms')}
              for stage in ('walk', 'descent')}
    gates = {
        'all_flights_complete': all(s['complete'] for s in summaries),
        'no_missing_or_exhausted_hits': all(s['exhausted_hits'] == s['loading_hits'] == s['not_ready_after_initial_load'] == 0 for s in summaries),
        'settled_cuts_match': cuts[0] == cuts[1],
        'arrival_improves': candidate['settlement_ms']['returned_ground'] < control['settlement_ms']['returned_ground'],
        'p95_within_15_percent': all(r['p95_ms'] <= 1.15 for r in ratios.values()),
        'max_within_2x': all(r['max_ms'] <= 2 for r in ratios.values()),
    }
    results[resolution] = {'gates': gates, 'ratios': ratios, 'cuts': dict(zip(names, cuts)),
                           'arrival_ms': {name: s['settlement_ms']['returned_ground'] for name, s in zip(names, summaries)}}
results['numeric_gates_pass'] = all(all(r['gates'].values()) for r in results.values())
(root / 'batch-trial-comparison.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
