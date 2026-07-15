# nebula-core

Foundation layer of the Nebula baking framework. Every other crate in the workspace depends on this one and only this one for shared types and traits.

---

## What lives here

| Module | Purpose |
|---|---|
| `context` | `BakeContext` — holds the wgpu `Device` + `Queue`; the entry point for all GPU work |
| `scene` | `SceneGeometry`, `BakeMesh`, `LightSource`, `AudioEmitter`, `MaterialDesc` — the scene description every baker reads |
| `traits` | `BakePass`, `BakeInput`, `BakeOutput`, `BakeSerializer` — the four trait seams the whole system is built on |
| `progress` | `ProgressReporter` + `NullReporter` — optional async progress callbacks for editor UI |
| `error` | `NebulaError` — one unified error type for the whole workspace |

---

## The trait design

Nebula is modelled after `serde`:

| serde concept | Nebula concept | Role |
|---|---|---|
| `Serialize` | `BakeInput` | Config struct that drives a baker |
| `Serializer` | `BakePass` | The baker itself; has one `execute` method |
| `Deserializer` | `BakeOutput` | The data produced by a baker |
| format crate (json, bincode …) | `BakeSerializer` | Persistence backend (binary `.nebula` or JSON) |

The key insight is that **every baker is interchangeable at the call site**:

```rust
let output = SomeBaker.execute(&scene, &config, &ctx, &reporter).await?;
```

You swap the baker, config, and output types — the surrounding plumbing stays the same.

---

## The GPU context

`BakeContext` is a thin `Arc<Device> + Arc<Queue>` wrapper. There are two ways to get one:

```rust
// Standalone headless — picks the best GPU on the system.
let ctx = BakeContext::new().await?;

// Share the renderer's device (no second GPU context, no extra VRAM).
let ctx = BakeContext::from_wgpu(device, queue, name, vendor, limits);
```

The second form is the preferred path when running inside the Helio editor. A single wgpu device can safely run both render frames and compute bakes; Nebula never allocates its own device unless you explicitly ask.

---

## The scene description

`SceneGeometry` is a plain data bag the editor fills from its runtime scene graph and hands to every baker:

- **`meshes`** — triangle soups with positions, normals, primary UVs, optional lightmap UVs, and a per-vertex material index.
- **`materials`** — PBR parameters (albedo, roughness, metallic, emissive) plus acoustic coefficients (absorption, scattering) used by `nebula-audio`.
- **`lights`** — directional, point, spot, area, and emissive-mesh variants. The `bake_enabled` flag lets you exclude dynamic-only lights from static bakes.
- **`audio_emitters`** — source positions and directivity used by the acoustic baker.
- **`sky_hdr`** — optional RGBE panorama for image-based lighting.

---

## How baking improves performance

Every baker in this workspace follows the same fundamental pattern:

1. **Bake time (offline):** run an expensive simulation — path tracing, ray casting, acoustic simulation — on the GPU. This can take seconds or minutes. The result is a static asset written to disk.
2. **Load time:** read the asset back. A lightmap is just a texture; a navmesh is just a list of polygons; an acoustic impulse response is just an array of floats.
3. **Runtime (every frame):** do the cheap version. Sample a texture. Look up a PVS bit. Read a reverb parameter. The expensive simulation never runs again.

The runtime result is **identical** to what you would get from a real-time simulation — because it *is* the output of a real simulation, frozen in time.

---

## License

MIT OR Apache-2.0
