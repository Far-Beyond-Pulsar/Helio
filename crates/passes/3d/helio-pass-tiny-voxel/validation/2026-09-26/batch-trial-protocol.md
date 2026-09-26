# Bounded dispatch trial

Recorded before the candidate flight comparison.

Control: Helio fcfb4d85, dispatch job cap 256, saved executable voxel_flight-before-batch.exe.
Candidate: cap 1024, independent exact-evaluation limits unchanged at 64 warm / 256 cold bricks, edit-reference limit unchanged at 262144. No terrain sampling, material, tree selection or publication changes.

Run the same 720p native and 1080p Quality flight for control and candidate, sequentially after the native compilation finishes. The existing Python GPU job stays untouched. Record GPU utilization before the comparison and label all times contended if it remains active.

Retain the candidate only if all unit/GPU and flight assertions pass, the settled cuts have matching brick counts and pixel budgets, the captures show no new visual failure, and returned-ground settlement improves at both resolutions. Walking/descent p95 may not exceed control by more than 15%, and their maximum frames may not exceed 2x control. No further parameter search in this trial if these gates fail.

These are engineering admission gates under workstation contention, not a claim of isolated performance qualification. The existing far-fidelity failure remains a failure even if dispatch scheduling improves.
