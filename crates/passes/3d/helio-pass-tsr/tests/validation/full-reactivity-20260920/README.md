# Full TSR reactivity contract

TsrPass::set_reactivity(1.0) is documented as current-frame-only, but compute_blend_factor subsequently mixed that weight back toward 0.5 according to frame delta. History leaked into the result even at full reactivity. The production resolve now uses its reset/current-only path when reactivity is at least one, preserving current linear depth and sharpening without sampling history. Default and intermediate reactivity are unchanged.

The existing full production-pipeline GPU test now compares full reactivity against reset at 120, 60 and 15 Hz, plus a 0.5-second frame. It fails before the change (readback 0.48632813 versus reset 0.48608398 in the first failing case), and passes after it. All five TSR GPU/unit tests pass: cargo test --release -p helio-pass-tsr --lib. Before/after output is retained. Release cathedral build passes.

The capture harness exposes HLFS_TSR_REACTIVITY in [0,1], requiring HLFS_TSR_NATIVE. Two 100-frame native-1440p moving-camera captures use RT presampling, stone materials, no SSR, native TSR: default reactivity zero versus one. The final images were inspected. Current-only output removes temporal accumulation globally and exposes more aliasing; it is a diagnostic control, not the recommended way to render glass. The change is an API correctness fix, not a claim that the remaining glass ghosting, shadow shimmer or 3-4 ms goal is solved. Per-pixel reactive coverage/depth for transparent surfaces remains absent.

The raw CSV covers HLFS only, not TSR or full-frame GPU time. No performance improvement is attributed to this branch. No new 4K stress claim.
