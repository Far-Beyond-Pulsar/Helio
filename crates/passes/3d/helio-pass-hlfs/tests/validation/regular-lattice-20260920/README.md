# Rejected regular-lattice reconstruction

Base fb6db84a with photographic cathedral materials. The capture harness now supports HLFS_CAPTURE_TAIL=N to save consecutive final frames. The default remains one. This reveals four-phase variation that the old 31/63/99 capture points (all the same phase) missed. These static scene tests use HLFS_FIXED_CAMERA=1, HLFS_CAPTURE_TAIL=4, 100 frames, native 1440p, FXAA, RT presampling, no SSR, default stone/normal maps. Reference adds HLFS_REFERENCE=1 and shades at full internal resolution. Frames 96-99 are retained.

The rejected patch replaces the spatially staggered sampling pattern with a globally regular lattice in presampled mode, aligns reconstruction weights to the actual phase origin, and adjusts previous-frame sample lookup. A first version applying this to all modes failed two legacy screen-space energy tests; the scoped version preserves legacy sampling and passes 2 unit, 14 screen-space GPU and 16 RT GPU tests. That test success was insufficient for real-scene acceptance.

At the fixed cathedral view, four-frame display RGB NRMSE relative to the full-resolution reference improved from 5.6024% to 4.9026%. Mean temporal per-channel standard deviation (8-bit display values) improved from 0.8063 to 0.7471; reference was 0.1126. Pixels with temporal range above 20 in any RGB channel fell from 3.0995% to 2.8955%. These are rendered-image measures, not linear-radiance error. Native 1440p HLFS-only median was 7.4235 ms control and 7.6119 ms candidate; single runs, no speedup established.

Moving cathedral NRMSE at frames 31/63/99 improved from 4.9325/5.4328/5.4488% to 4.3431/4.8731/4.9106%. Baseline images are the receiver-offset experiment's candidate-2 captures. Candidate and full-resolution references are saved locally and final frames retained here.

The 1024-light technology scene rejected the change. At a fixed camera position 0.5, native 640x360 diagnostic output, 100-frame candidate/control runs compared to a 17-frame full-resolution reference at the same camera position: errors at frames 31/63/99 worsened from 22.1179/22.3804/22.6149% to 23.0922/23.6876/23.8073%. Mean display brightness at frame 99 relative to reference fell from 0.9374 to 0.9248. The candidate image also shows a stronger structured sampling pattern on the floor. This diagnostic is not a 1440p performance acceptance run.

Production shaders were restored. The patch remains only as rejected evidence, not an optional production mode. Matching actual sample positions helps the cathedral's exact-light tier, but globally regular placement damages the stochastic many-light tier. Future reconstruction must retain decorrelation and be tested on both tiers. No artifact-free or 3-4 ms acceptance is claimed.

Restoration check: rebuilt the production cathedral and repeated the fixed-camera run. Frames 96-99 exactly match the original baseline in all RGBA channels.
