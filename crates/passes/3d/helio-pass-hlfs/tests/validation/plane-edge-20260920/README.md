# Rejected receiver-plane reconstruction expansion

The trial allowed a neighbor rejected by view-depth difference when its actual G-buffer position was close to the current receiver plane and normals aligned. Native 1440p fixed-camera cathedral captures were compared with exhaustive full-resolution lighting. The baseline/reference are preserved in the adjacent `sparse-edge-20260920` report.

Reference RGB NRMSE worsened from 5.2557% to 5.3784%; bottom-440-row error worsened from 3.7936% to 3.8825%; mean RGB temporal standard deviation across frames 96–99 rose from 0.98744 to 1.01018 (8-bit units). HLFS-only median increased from 7.7133 to 8.1531 ms in these single sequential runs. The candidate was visually inspected and reverted. `rejected.patch` and its frame-99 capture preserve the attempted change, not production behavior.

A subsequent radiance-contrast experiment, gated on at most 32 global light slots and exact-neighbor metadata, produced pixel-identical output for this scene. It was also removed; no benefit is claimed. Investigating that ineffective gate led to the separate sparse light-allocation regression and fix.
