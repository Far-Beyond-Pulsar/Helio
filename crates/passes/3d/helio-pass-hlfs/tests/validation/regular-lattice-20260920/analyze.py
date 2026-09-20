from pathlib import Path
import json
import numpy as np
from PIL import Image
root = Path(__file__).resolve().parent
def rgb(folder, file):
    return np.asarray(Image.open(root / folder / file))[..., :3].astype(float)
def error(a, b):
    return float(100 * np.sqrt(np.mean((a-b)**2) / np.mean(b**2)))
reference = np.stack([rgb('phase-static-reference', f'cathedral-reference-{i:03}.png') for i in range(96,100)])
results = {'reference_temporal_std': float(reference.std(axis=0).mean())}
for name in ['phase-static-baseline', 'phase-static-candidate']:
    frames = np.stack([rgb(name, f'cathedral-{i:03}.png') for i in range(96,100)])
    results[name] = {'nrmse_percent': error(frames, reference),
        'mean_temporal_std': float(frames.std(axis=0).mean()),
        'fraction_range_over_20': float(np.mean(np.max(np.ptp(frames,axis=0),axis=2)>20))}
reference = rgb('phase-tech-reference','technology-reference-016.png')
for name in ['phase-tech-control','phase-tech-candidate']:
    frame = rgb(name, 'technology-099.png')
    results[name] = {'nrmse_percent': error(frame,reference), 'mean_ratio': float(frame.mean()/reference.mean())}
print(json.dumps(results,indent=2))
