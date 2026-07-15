# nebula-gpu

Thin, strongly-typed wgpu helpers used internally by every Nebula baker. You can also use these utilities directly when writing your own custom bake passes.

---

## What lives here

| Module | Types | Purpose |
|---|---|---|
| `buffer` | `UniformBuffer<T>`, `StorageBuffer<T>` | Type-safe GPU buffer wrappers; no raw `bytes_of` at call sites |
| `texture` | `BakeTexture`, `BakeTextureArray`, `TextureFormat2D` | 2-D and array textures with built-in CPU readback |
| `compute` | `ComputePipeline`, `ComputePass` | Pipeline template helpers that reduce bind-group boilerplate |
| `readback` | `GpuReadback` | Staging-buffer helper for `COPY_SRC → MAP_READ` readbacks |

### Global constants

```rust
pub const MAX_TEXTURE_DIM: u32 = 8192;   // All bakers clamp to this.
pub const WORKGROUP_SIZE:  u32 = 8;      // Standard 8×8 tile for 2-D compute.
```

---

## Design philosophy

Every baker crate (nebula-light, nebula-ao, …) could write raw wgpu from scratch, but that means duplicating the same staging-buffer logic, the same `copy_texture_to_buffer` alignment math, and the same `poll(wait_indefinitely)` dance in every place. `nebula-gpu` factors all of that out once.

### Typed buffers

```rust
// Instead of: device.create_buffer_init(...) and bytemuck everywhere
let params = UniformBuffer::new(&ctx.device, "my_params", &my_struct);
// params.group  → the ready-to-bind BindGroup
// params.write(queue, &new_value) → update without re-creating
```

### BakeTexture and built-in readback

```rust
let tex = BakeTexture::new(
    &ctx.device, "nebula_lightmap",
    1024, 1024, TextureFormat2D::RGBA32F, 1,
    wgpu::TextureUsages::empty(),
);

// ... dispatch compute passes that write into tex.view ...

let bytes: Vec<u8> = tex.read_back(&ctx.device, &ctx.queue);
// bytes is row-major, no padding — ready to hand to the serializer.
```

The `read_back` method handles:
- Allocating an aligned staging buffer (wgpu requires 256-byte row alignment internally; the output strips padding so callers get a clean byte slice).
- Submitting a `copy_texture_to_buffer` command.
- Mapping and waiting synchronously — acceptable for offline baking.

---

## Workgroup sizing

All 2-D compute shaders in the workspace use `8×8` tiles (= 64 threads, one warp/wavefront). Dispatches are always rounded up:

```rust
pass.dispatch_workgroups(
    res.div_ceil(WORKGROUP_SIZE),
    res.div_ceil(WORKGROUP_SIZE),
    1,
);
```

This means a 1024×1024 lightmap dispatches `128×128` workgroups = 16 384 workgroups × 64 threads = ~1 M threads in flight simultaneously on the GPU — the same parallelism that makes baking fast enough to be practical.

---

## When to use this crate directly

- You are writing a **custom bake pass** (your own `impl BakePass`) and want the same GPU conveniences the built-in bakers use.
- You need a compute pipeline that writes into a texture or a large storage buffer and reads it back.

If you only want the high-level baking API, depend on `nebula` (the façade) and ignore this crate entirely.

---

## License

MIT OR Apache-2.0
