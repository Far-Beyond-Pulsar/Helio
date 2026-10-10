# Recording cache

Helio#330, part 5. On by default; `HELIO_RECORDING_CACHE=0` (or `off`,
`false`) switches it off, as does `RenderGraph::set_recording_cache(false)`.

## What it does

Every frame, each cacheable unit records its commands into a stream
(`helio_core::cmd` in recorded mode) instead of a wgpu encoder. The graph
compares the stream with the recordings it cached for that unit:

- **Hit**: the cached command buffers are submitted again. wgpu does not
  encode or validate them; it only adds the per-submission barriers and keeps
  the resources alive.
- **Miss**: the stream is encoded once into reusable command buffers and
  cached. Each unit keeps up to four recordings, so ping-pong passes alternate
  between two. A recording unused for 16 frames is dropped.

Recordings compare by resource identity: the same pipelines, bind groups,
buffers and views, the same arguments. A unit that records anything different
misses, so correctness never depends on a pass declaring what it uses. Data a
pass changes through `queue.write_buffer` is not part of the recording, which
is why uniform uploads don't cause misses.

Passes need no changes. `prepare()`, `execute()` and `publish()` still run
every frame in graph order; only encoding is skipped. A resubmitted command
buffer is validated like a new one, so a buffer it copies into must not be
mapped, exactly as when encoding every frame.

GPU timing is recorded like any other command. The profiler's query indices
restart every frame, so a frame of the same shape writes the same timestamps.

Resubmitting command buffers needs the reusable command buffers in
[Far-Beyond-Pulsar/wgpu](https://github.com/Far-Beyond-Pulsar/wgpu/tree/helio/v30.0.1-reusable-command-buffers),
which the workspace patches in. Vulkan and D3D12 support it; on other backends
the cache switches itself off after the first frame.

## Units

A unit is one of:

- a pass outside a fused render-pass chain;
- a pass with a prebuilt render bundle (the bundle replays in its render pass,
  as when encoding directly);
- a whole fused chain, recorded as its one render pass, with chain-transparent
  passes bridged into it recording on the compute stream.

A unit is recorded directly instead when:

- one of its passes returns `false` from `RenderPass::supports_recording_cache`
  (none do in Helio);
- wgpu could not make its command buffers reusable (acceleration structure
  builds and ray queries, surface textures, deferred actions); it is then
  recorded directly until the graph is rebuilt;
- it missed 8 frames in a row: a miss costs more than recording directly (the
  stream is compared, then replayed into an encoder finished on the render
  thread), so it is recorded directly for the next 64 frames, then tried
  again.

The whole cache is off while the finish breakdown (`HELIO_FINISH_BREAKDOWN`)
is on.

Consecutive units submit their own command buffers back to back. Directly
recorded passes before a unit are cut into a command buffer of their own, but
only when the encoder holds something.

## Seeing what it does

`RenderGraph::recording_cache_stats()` (and `Renderer::recording_cache_stats()`)
list each unit's hits, misses, frames recorded directly while backing off, and
number of recordings, plus every pass recorded directly last frame and why.
For the last miss they also say where its stream first differed from the
previous recording, for example
`graphics stream: command 3 (SetBindGroup) changed its arguments`.
`move_benchmark` prints this summary after each row.

## Measured

`move_benchmark --graph default --editor`, lavapipe (4 cores), 320x180, 64
lights, three alternating rounds with the cache off and on (median CPU
`render` per row, range over the rounds):

| objects | workload | off | on | hit rate |
|---:|---|---:|---:|---:|
| 1k | idle | 1.53–1.58 ms | 1.37–1.52 ms | 97% |
| 1k | move_one | 2.46–2.81 ms | 1.92–2.10 ms | 98% |
| 1k | move_lights | 1.62–1.70 ms | 1.42–1.45 ms | 99% |
| 16k | idle | 1.83–1.99 ms | 1.54–1.58 ms | 97% |
| 16k | move_one | 2.84–2.96 ms | 2.09–2.15 ms | 98% |
| 16k | move_lights | 1.73–1.88 ms | 1.53–1.57 ms | 99% |

The hit rates count from the first frame; almost every miss is in the first
two frames. Every pass in the editor graph is cached, fused chains included.
With the `hlfs` graph, HLFS's ray-traced mode uses acceleration structures and
is recorded directly.

Frames rendered with the cache on are byte-identical to frames rendered with
it off (`move_benchmark --dump-frame`) for every row above, and for the
`hlfs` graph, screen-space and ray-traced.

lavapipe's GPU time also drops with the cache on (16k objects: about 2.3 s
to 1.3 s per frame). lavapipe runs on the same four cores, so this is most
likely less CPU contention from wgpu's encoding; don't expect it on a real GPU.

## Limits

- Ray-traced passes that use acceleration structures can't be made reusable.
- Cached recordings keep the resources they use alive until they are evicted
  (16 unused frames), so a resize frees the old render targets that much
  later.
