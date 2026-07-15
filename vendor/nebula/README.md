# Nebula

**Nebula** is a modular, GPU-accelerated data-baking framework for the Helio engine.  
It handles every kind of pre-computed static data — lightmaps, ambient occlusion, reflection/irradiance probes, spatial audio, potentially-visible sets, and navigation meshes — in a single, coherent, composable system.

---

## Design philosophy

Nebula is deliberately designed after the serde model:

| serde concept          | Nebula concept           |
|------------------------|--------------------------|
| `Serialize`            | `BakeInput`              |
| `Deserializer`         | `BakePass`               |
| `Serializer`           | `BakeSerializer`         |
| format crate (json, …) | baker crate (light, ao, …) |

Each baker crate is **independently useable**. You pull in only what you need. The `nebula` façade re-exports everything.

---

## Crate family

| Crate | Purpose |
|---|---|
| `nebula-core` | Core traits (`BakePass`, `BakeInput`, `BakeOutput`), the GPU `BakeContext`, scene types, progress reporting, serializer traits |
| `nebula-gpu` | Thin wgpu helpers: typed buffers, textures, compute pipeline templates, GPU→CPU readback |
| `nebula-serialize` | Two serialization backends: a compact binary `.nebula` chunked format (with optional zstd compression) and a JSON metadata format |
| `nebula-light` | GPU path-traced **lightmap baking** — direct illumination, multi-bounce GI, area-light support |
| `nebula-ao` | GPU hemisphere-sampled **ambient occlusion** baking |
| `nebula-probe` | **Reflection & irradiance probe** baking — cubemap capture, specular pre-filter, diffuse convolution |
| `nebula-audio` | GPU-accelerated **geometric acoustics** — room impulse responses (RIR), reverb-zone parameters, sound occlusion precompute |
| `nebula-visibility` | **Potentially-visible set (PVS)** baking via GPU visibility testing |
| `nebula-nav` | **Navigation mesh** construction and pathfinding pre-computation |
| `nebula` | Façade: re-exports everything, provides `BakeContext::new()` shortcut |

---

## Quick start

```toml
[dependencies]
nebula = { path = "../Nebula/crates/nebula" }
```

```rust
use nebula::{BakeContext, SceneGeometry};
use nebula::light::{LightmapBaker, LightmapConfig};
use nebula::serialize::binary::NebulaBinarySerializer;

let ctx = BakeContext::new().await?;

let baker  = LightmapBaker::default();
let config = LightmapConfig { resolution: 1024, bounce_count: 3, ..Default::default() };
let output = baker.execute(&scene, &config, &ctx, &nebula::NullReporter).await?;

let ser = NebulaBinarySerializer::default();
let mut file = std::fs::File::create("scene.lightmap.nebula")?;
ser.serialize(&output, &mut file)?;
```

---

## GPU requirements

Nebula uses the same wgpu fork as the Helio renderer. The baking context can either create its own headless wgpu device (for offline tools) **or** share the renderer's existing `(Device, Queue)` pair (for in-editor baking with no device duplication).

```rust
// Share the renderer's device:
let ctx = BakeContext::from_wgpu(device.clone(), queue.clone(), adapter_info);
```

---

## File format

Nebula uses a chunked binary format (`.nebula`) broadly inspired by RIFF/IFF:

```
[MAGIC: "NEBULA\0\0"] [version: u32]
[CHUNK_TYPE: u32] [chunk_len: u64] [data: bytes]
[CHUNK_TYPE: u32] [chunk_len: u64] [data: bytes]
...
[CHUNK_TYPE: END] [chunk_len: 0]
```

Each chunk is independently zstd-compressed (optional, always indicated by a flags field). This allows streaming partial reads and future extensibility without breaking older readers.

---

## License

MIT OR Apache-2.0
