# Voxel planet validation tools

This directory contains CPU-only capture and HDR/TSR analysis helpers for the
stored tiny-voxel terrain pass. The runtime implementation lives in
`crates/passes/3d/helio-pass-tiny-voxel`; the current Helio 3.0 graph remains
SceneDB-based, so the old standalone viewer was removed instead of carrying a
second legacy scene integration.

The tests are directly runnable without the Helio workspace dependencies:

```powershell
python tools/voxel-planet/test_linear_hdr_reference.py
python tools/voxel-planet/test_motion_hdr_reference.py
python tools/voxel-planet/test_tsr_diagnostics.py
```

The analysis scripts consume captures produced by the active renderer and
report unresolved pixels, timing percentiles, image differences, HDR reference
accumulation, and TSR history diagnostics. They do not claim playable frame
rate; GPU timings must be measured with the renderer's own warmed frame probes.
