# Rejected zero-incident branch, 2026-09-23

I briefly returned zero `Lighting` from `evaluate_incident` when `N·L == 0`
or incident radiance was zero. The arithmetic result should be zero in these
cases; this experiment tested whether avoiding the remaining BRDF instructions
would help the native 2560 × 1440, 1,024-static-light gallery.

In one release pilot on the RTX 3060, sampling GPU median rose from the
control's 9.55 ms to 10.12 ms. HLFS rose from 14.73 to 15.40 ms and the full
render graph from 21.03 to 21.76 ms. These are single, non-interleaved captures,
so they screen out this candidate rather than establish a precise effect size.
Frame 0 was byte-identical; later stochastic frames differed. The branch was
removed. The candidate's configuration and raw GPU timings are retained here;
the control is the earlier unchanged gallery capture.
