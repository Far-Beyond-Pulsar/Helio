# Recording cache

Helio#330, part 5. Off by default while it is brought up; switch it on with
`HELIO_RECORDING_CACHE=1` or `RenderGraph::set_recording_cache(true)`.

## What it does

Every frame, each cacheable pass records its commands into a stream
(`helio_core::cmd` in recorded mode) instead of a wgpu encoder. The graph
compares the stream with the recordings it cached for that pass:

- **Hit**: the cached command buffers are submitted again. wgpu does not
  encode or validate them; it only adds the per-submission barriers and keeps
  the resources alive.
- **Miss**: the stream is encoded once into reusable command buffers and
  cached. Each pass keeps up to four recordings, so ping-pong passes alternate
  between two. A recording unused for 16 frames is dropped.

Recordings compare by resource identity: the same pipelines, bind groups,
buffers and views, the same arguments. A pass that records anything different
misses, so correctness never depends on a pass declaring what it uses. Data a
pass changes through `queue.write_buffer` is not part of the recording, which
is why uniform uploads don't cause misses.

GPU timing is recorded like any other command. The profiler's query indices
restart every frame, so a frame of the same shape writes the same timestamps.

Resubmitting command buffers needs the reusable command buffers in
[Far-Beyond-Pulsar/wgpu](https://github.com/Far-Beyond-Pulsar/wgpu/tree/helio/v30.0.1-reusable-command-buffers),
which the workspace patches in. Vulkan and D3D12 support it; on other backends
the cache switches itself off.

## What is cached

A pass is cached when it:

- is not part of a fused render-pass chain,
- has no prebuilt render bundle,
- returns `true` from `RenderPass::supports_recording_cache` (the default).

`ShadowMatrixPass` returns `false`: it copies into a buffer it maps
asynchronously, and replaying the copy can race the map.

A pass whose command buffers wgpu cannot make reusable (acceleration
structure builds, surface textures, deferred actions) is recorded straight into
wgpu from then on. The whole cache is off while the finish breakdown
(`HELIO_FINISH_BREAKDOWN`) is on.

## Seeing what it does

`RenderGraph::recording_cache_stats()` (and `Renderer::recording_cache_stats()`)
list each cached pass's hits, misses and number of recordings. For the last
miss they also say where its stream first differed from the previous recording,
for example `graphics stream: command 3 (SetBindGroup) changed its arguments`.
`move_benchmark` prints this summary after each row when the cache is on.

## Measured

`move_benchmark --graph default --editor`, lavapipe (4 cores), 320x180,
alternating runs with the cache off and on:

| objects | workload | CPU `render`, off | CPU `render`, on | hit rate |
|---:|---|---:|---:|---:|
| 1k | idle | 1.4–1.7 ms | 1.4–1.5 ms | 95% |
| 1k | move_one | 2.7–3.0 ms | 2.1–2.2 ms | 97% |
| 16k | idle | 1.7–1.8 ms | 1.7 ms | 95% |
| 16k | move_one | 2.6–2.8 ms | 2.0–2.2 ms | 97% |

Almost all misses are the first two frames. Frames rendered with the cache on
are byte-identical to frames rendered with it off (`move_benchmark
--dump-frame`). On lavapipe the GPU time also dropped by about a third with the
cache on; that is not explained yet, so treat it as an observation.

## Not yet

- Fused chains and render-bundle passes are always recorded directly.
- A cached pass cuts the compute and graphics streams around itself, which
  adds a command buffer per cut.
- Pulsar-Native's root manifest needs the same `[patch.crates-io]` section
  before its Helio pin moves past this change.
