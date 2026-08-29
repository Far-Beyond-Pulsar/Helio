# Helio controls

`helio-controls` is the reusable input-to-camera layer for Helio applications.
It deliberately sits between platform routing and rendering:

```text
winit / TRUEOS UI4 / OpenXR / automation
                 │
                 ▼
       NavigationAction + NavigationState
                 │
                 ▼
              FlyCamera ───► helio::Camera::from_fly
```

The semantic core knows no window, cursor, device, or compositor type. Platform
adapters decide which user/device route has focus and translate held controls
plus relative look motion. `FlyCameraConfig` retains the meaningful variations
found in existing examples: view-plane versus ground-plane motion, normalized
or legacy diagonal speed, pitch bounds, boost, sensitivity, and frame-delta
clamping.

Enable the `winit` feature for the small desktop adapter. TRUEOS uses its UI4
`input_routes` snapshot directly so multi-mouse/multi-keyboard focus and pairing
remain compositor policy rather than becoming engine-global state.
