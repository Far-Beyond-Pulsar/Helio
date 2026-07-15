# nebula-serialize

Two serialization backends for Nebula bake outputs. Both implement `nebula_core::traits::BakeSerializer` so they are interchangeable anywhere in the API.

---

## Backends

### `NebulaBinarySerializer` — compact binary `.nebula` format

The primary production format. A `.nebula` file is a chunked binary container inspired by RIFF/IFF:

```
[MAGIC: "NEBULA\0\0"] [version: u32]
[CHUNK_TYPE: u32] [chunk_len: u64] [data: bytes]
[CHUNK_TYPE: u32] [chunk_len: u64] [data: bytes]
...
[CHUNK_TYPE: END]  [chunk_len: 0]
```

Each chunk is independently zstd-compressed (controlled per-serializer by the `Compression` enum). The flags field inside each chunk header records whether compression was applied, so mixed files are always readable.

**Chunk tags** are four-byte ASCII codes defined in the baker crate that owns that data type (never in this crate):

| Tag | Baker | Data |
|---|---|---|
| `LMAP` | nebula-light | Lightmap atlas texels + atlas regions |
| `AAOC` | nebula-ao | Ambient occlusion R32F texels |
| `REFL` | nebula-probe | Specular cubemap face data |
| `IRRD` | nebula-probe | Irradiance SH coefficients |
| `AUIR` | nebula-audio | Room impulse responses + reverb zones |
| `PVSD` | nebula-visibility | PVS bit matrix |
| `NAVD` | nebula-nav | Navigation polygon mesh |

Independent chunks allow:
- **Streaming partial reads** — load only the chunk(s) you need.
- **Incremental re-baking** — replace one chunk without rewriting the whole file.
- **Forward compatibility** — unknown chunk types are skipped by older readers.

#### Compression levels

```rust
pub enum Compression {
    None,
    Fast,       // zstd level 1  — ~2× size reduction, very fast
    Balanced,   // zstd level 9  — good ratio, moderate speed (default)
    Best,       // zstd level 19 — maximum ratio, slow
}
```

Texel-heavy data (lightmaps, AO) compresses well with `Fast`. Floating-point audio IRs benefit from `Balanced` or `Best`.

### `NebulaJsonSerializer` — human-readable JSON metadata

Writes the bake configuration and scalar parameters as readable JSON. Useful for:
- Diffing bake parameters in version control (`git diff scene.lightmap.json`).
- Embedding bake settings in editor project files.
- Inspecting outputs in external tools without writing a binary parser.

The JSON backend is deliberately **not** used for bulk texel data — it is paired with the binary format, not a replacement for it.

---

## Usage

```rust
use nebula_serialize::{NebulaBinarySerializer, Compression};

let ser = NebulaBinarySerializer { compression: Compression::Balanced };
let mut file = std::fs::File::create("scene.lightmap.nebula")?;
ser.serialize(&lightmap_output, &mut file)?;
```

```rust
// Round-trip
let mut file = std::fs::File::open("scene.lightmap.nebula")?;
let output: LightmapOutput = ser.deserialize(&mut file)?;
```

---

## How baking improves performance — the serialization angle

The `.nebula` file is the boundary between the expensive offline simulation and the cheap runtime lookup. Without a compact, streamable format:

- The baked data would have to be recomputed at startup (unacceptable) or stored in an inefficient ad-hoc format (wastes disk and memory bandwidth).
- zstd compression on a typical 1024² lightmap reduces 16 MB of RGBA32F data to ~3–4 MB on disk and decompresses in milliseconds — well within level-load budgets.
- Independent chunks mean the runtime only loads what it needs. A mobile build can skip the full-precision HDR lightmap and load the half-float version; a server build can skip all rendering data entirely and load only navmesh and PVS.

---

## License

MIT OR Apache-2.0
