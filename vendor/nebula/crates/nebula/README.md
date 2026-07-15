# nebula

Façade crate — re-exports every Nebula sub-crate behind a single dependency, with feature flags to pull in only what you need.

---

## Quick start

```toml
[dependencies]
nebula = { path = "../Nebula/crates/nebula" }
```

By default all bakers are enabled. To minimise compile time and binary size, disable unwanted features:

```toml
[dependencies]
nebula = { path = "../Nebula/crates/nebula", default-features = false, features = ["light", "ao"] }
```

---

## Feature flags

| Feature | Crate enabled | What it brings in |
|---|---|---|
| `light` | `nebula-light` | `nebula::light::{LightmapBaker, LightmapConfig, LightmapOutput}` |
| `ao` | `nebula-ao` | `nebula::ao::{AoBaker, AoConfig, AoOutput}` |
| `probe` | `nebula-probe` | `nebula::probe::{ProbeBaker, ProbeConfig, ReflectionOutput, IrradianceOutput}` |
| `audio` | `nebula-audio` | `nebula::audio::{AcousticBaker, AcousticConfig, AcousticOutput}` |
| `visibility` | `nebula-visibility` | `nebula::visibility::{PvsBaker, PvsConfig, PvsOutput}` |
| `nav` | `nebula-nav` | `nebula::nav::{NavBaker, NavConfig, NavOutput}` |

`nebula-core`, `nebula-gpu`, and `nebula-serialize` are always included — they are the foundation every baker builds on.

---

## Complete example

```rust
use nebula::prelude::*;
use nebula::light::{LightmapBaker, LightmapConfig};
use nebula::ao::{AoBaker, AoConfig};
use nebula::audio::{AcousticBaker, AcousticConfig, ListenerPoint};
use nebula::nav::{NavBaker, NavConfig};
use nebula::serialize::NebulaBinarySerializer;

#[pollster::main]
async fn main() -> nebula::core::Result<()> {
    // Create (or share) a GPU context.
    let ctx = BakeContext::new().await?;

    // Build a scene description from your runtime scene graph.
    let scene = build_scene();

    // ── Lighting ──────────────────────────────────────────────────────────────
    let lightmap = LightmapBaker.execute(
        &scene,
        &LightmapConfig { bounce_count: 2, ..Default::default() },
        &ctx,
        &NullReporter,
    ).await?;

    // ── Ambient occlusion ─────────────────────────────────────────────────────
    let ao = AoBaker.execute(&scene, &AoConfig::default(), &ctx, &NullReporter).await?;

    // ── Acoustics ─────────────────────────────────────────────────────────────
    let acoustic_cfg = AcousticConfig {
        listener_points: vec![
            ListenerPoint { position: [0.0, 1.5, 0.0], label: Some("spawn".into()) },
        ],
        ..Default::default()
    };
    let acoustics = AcousticBaker.execute(&scene, &acoustic_cfg, &ctx, &NullReporter).await?;

    // ── Navigation ────────────────────────────────────────────────────────────
    let nav = NavBaker.execute(&scene, &NavConfig::default(), &ctx, &NullReporter).await?;

    // ── Serialise ─────────────────────────────────────────────────────────────
    let ser = NebulaBinarySerializer::default();
    ser.serialize(&lightmap, &mut std::fs::File::create("scene.lightmap.nebula")?)?;
    ser.serialize(&ao,       &mut std::fs::File::create("scene.ao.nebula")?)?;
    ser.serialize(&acoustics, &mut std::fs::File::create("scene.acoustic.nebula")?)?;
    ser.serialize(&nav,      &mut std::fs::File::create("scene.nav.nebula")?)?;

    println!("All bakes complete.");
    Ok(())
}
```

---

## The baking mental model

Think of Nebula as a **one-time rendering pass that runs in the editor or a build pipeline**, not at game startup:

```
Scene data (meshes, lights, materials, audio emitters)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  Nebula bake (seconds to minutes, runs once)        │
│                                                     │
│  nebula-light     →  lightmap atlas (.nebula)       │
│  nebula-ao        →  AO texture (.nebula)           │
│  nebula-probe     →  reflection cubemaps (.nebula)  │
│  nebula-audio     →  impulse responses (.nebula)    │
│  nebula-visibility →  PVS bit matrix (.nebula)      │
│  nebula-nav       →  navmesh polygons (.nebula)     │
└─────────────────────────────────────────────────────┘
          │
          ▼  (load at startup, ~milliseconds)
┌─────────────────────────────────────────────────────┐
│  Runtime (every frame, near-zero cost)              │
│                                                     │
│  Texture sample  ←  lightmap, AO, probes           │
│  Bit lookup      ←  PVS culling                    │
│  Convolution     ←  acoustic IR playback           │
│  A* pathfinding  ←  navmesh graph traversal        │
└─────────────────────────────────────────────────────┘
```

Every baked file is the **frozen output of an expensive simulation** that would otherwise have to run in real time. The runtime result is identical — the only difference is *when* the computation happened.

---

## Prelude

```rust
use nebula::prelude::*;
// Brings in: BakeContext, NebulaError, NullReporter, ProgressReporter,
//            SceneGeometry, BakeInput, BakeOutput, BakePass, ChunkTag
```

---

## License

MIT OR Apache-2.0
