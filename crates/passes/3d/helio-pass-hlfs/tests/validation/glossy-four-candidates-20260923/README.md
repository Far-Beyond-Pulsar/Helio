# Rejected four-candidate glossy scan, 2026-09-23

After the [full-support GPU light proposal](../glossy-full-support-20260923/README.md),
I temporarily lowered the glossy minimum from eight to four candidates per
sample. The four glossy samples, exact selected-light BRDF, hardware ray query,
geometry, and native shading resolution remained unchanged. The source change
was one constant in `sample.wgsl` and was reverted after the quality check.

The fixed seed-11, 96-frame RT quality fixture used setting `1:8`, native
sample scale, 1,024 moving colored lights, glossy camera motion, a moving
blocker, key-light switch, reactive history, direct-only output, and motion
capture. Its existing spatial grain limit is 0.005. The four-candidate result
failed at frame 80 (0.01003) and frame 81 (0.00575); the accepted eight-candidate
result recorded 0.00421 at both frames with the same fixture. The
[candidate frame 80](candidate-f080.png) and
[frame 81](candidate-f081.png), [grain CSV](grain-metrics.csv),
[final-output CSV](quality.csv), and [motion CSV](motion-metrics.csv) preserve
the failure. The candidate's motion
metric passed, but that does not override the grain failure. No speed claim was
made because this variant was rejected before a timing capture.

The full consecutive candidate sequence remains in the ignored crate-local
`target/validation/glossy-four-candidate-seed11` directory on this machine.
