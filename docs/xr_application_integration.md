# Helio application XR integration

Native Helio applications can use one graphics path for desktop and OpenXR.
The application still creates and owns its window/event loop. Helio selects
the graphics device before the app creates GPU-backed SceneDB data, and then
routes the usual window frame to desktop rendering or headset rendering with a
PC mirror.

## Enable XR discovery

```rust,ignore
let graphics = HelioGraphicsContext::new(
    window.clone(),
    window.inner_size().width,
    window.inner_size().height,
    HelioGraphicsConfig {
        vr_support: true,
        application_name: "My Helio App".into(),
        ..Default::default()
    },
)
.await?;

// Build the SceneDB GPU mirror from these exact shared handles.
let device = graphics.device.clone();
let queue = graphics.queue.clone();
let scene = create_scene_with_gpu_mirror(device.clone(), queue.clone());

// Keep the app's normal renderer setup and scene graph builder.
let config = RendererConfig::new(
    window.inner_size().width,
    window.inner_size().height,
    graphics.surface_config.format,
)
.with_render_mode(RenderMode::Deferred);
let builder = RendererBuilder::new(config, scene.gpu_mirror_handle())
    .with_pass_build_context(/* the application's existing graph inputs */);

// Headset/session setup is automatic. This is None in desktop fallback mode;
// use it only if the app needs controller input or custom locomotion.
let (mut renderer, xr_input) = graphics.build_renderer(builder, config)?;
if let Some(input) = &xr_input {
    let controllers = input.controller_state()?;
    // Feed `controllers` into app-specific movement/actions.
}

// In the existing event loop:
graphics.render_frame(&mut renderer, &camera)?;
```

With `vr_support: true`, Helio attempts OpenXR startup first. When the runtime
is absent or startup fails, it logs the reason and creates a regular desktop
WGPU device and surface. `renderer_config` selects the runtime's eye resolution
and swapchain format for XR, or the window dimensions and format for desktop,
before the graph is built.
The renderer remains responsible for submitting the headset frame and
compositing its side-by-side PC mirror; the app does not build a stereo window
output.

The shared XR camera uses headset tracking automatically. Apps that require
custom locomotion or controller-driven interaction consume the optional
`HelioXrInput` returned by `build_renderer`; those behaviors remain app-specific.
Call `graphics.resize(&mut renderer, width, height)` when the app window changes
size. This updates the renderer resolution in desktop mode and keeps the eye
render size fixed to the runtime in XR mode.

## Existing externally managed graphics devices

Apps that already own device creation can keep using `XrRuntime::start_helio`,
`RendererConfig::for_xr`, and `Renderer::install_xr_runtime`. The high-level
`vr_support` option cannot replace a device after creation, so the shared
graphics context must be selected before creating GPU-backed scene data.

## Scope

This API is native-only because OpenXR's Vulkan session is native-only. Wasm
apps continue using their existing WebGPU setup. `RendererConfig::enable_xr`
is a low-level render-graph multiview option; it is separate from headset
discovery and is not needed by the context's dual-pass stereo path.
