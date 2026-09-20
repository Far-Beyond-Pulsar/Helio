# Rejected bracketing reconstruction

This trial preserved the original staggered sample placement and changed only presampled composition. It found the two actual sample columns bracketing a pixel, interpolated their row positions and used those coordinates for four-neighbour bilinear weights. It left temporal sample lookup and the sampling shader unchanged. See rejected.patch; production composite was restored.

Same fixed-view controls as ../regular-lattice-20260920: native 1440p cathedral, RT presampling, FXAA, normal-mapped stone, no SSR, 100 frames with final four phases saved. Four-frame NRMSE barely changed (5.6024% to 5.5913%) while mean temporal RGB standard deviation rose from 0.8063 to 0.9040. Composite median rose from 2.3194 to 2.6317 ms; HLFS-only median from 7.4235 to 7.8925 ms (single runs). At native 640x360, fixed technology camera 0.5 with 1024 lights, final-frame error worsened from 22.6149% to 23.6611% against the same full-resolution reference. These are display-RGB diagnostics, not linear radiometry or 1440p technology acceptance.

Rejected before further regression expansion: the scene controls already contradict the intended quality/performance improvement. The restored production HLFS shaders were used for subsequent successful native-1440p TSR captures. No reconstruction change is kept from this experiment.
