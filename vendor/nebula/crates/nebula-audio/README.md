# nebula-audio

GPU-accelerated **geometric acoustics** baking — room impulse responses (RIR), reverb-zone parameters, and sound occlusion pre-computation.

---

## For the audio engineer: what problem are we solving?

If you have worked in a DAW you already know the core concept. You are already using it every time you load a convolution reverb.

### Convolution reverb in the studio

When you load an impulse response (IR) of the Hagia Sophia into your reverb plugin and print a vocal through it, what is actually happening?

1. Someone went to that building with a starter pistol (or a sine sweep), fired it, and recorded the resulting sound with a microphone.
2. The recording — called a **Room Impulse Response (RIR)** — captures everything the room did to that impulse: the first reflection off the nearest wall (arriving a few milliseconds after the direct sound), the dense cloud of late reverb building up over hundreds of milliseconds, and the gradual frequency-dependent decay as low-frequency energy hangs around longer than high-frequency energy (absorbed by air and soft surfaces).
3. Your convolution reverb plugin performs a mathematical convolution between your dry vocal and that recorded IR. The result is exactly what your vocal would have sounded like in that building.

The IR contains the complete acoustic fingerprint of the space. The convolution is the mechanism that applies it.

### The game audio problem

In a game, the player moves through many different acoustic spaces: a small tiled bathroom, a vast stone cathedral, a concrete car park, an outdoor forest clearing. Each sounds completely different.

In a DAW you have the luxury of picking one reverb preset per track. In a game the "room" changes continuously as the player walks around.

The naive solution — simulate the room acoustics in real time from first principles — is computationally prohibitive:

- **Ray tracing is expensive.** Getting a convincing reverb tail requires thousands of acoustic ray paths per listener position. Even on a modern GPU this is a dedicated full-frame workload.
- **Convolution reverb is expensive.** Convolving an audio stream with a 2-second IR (88 200 samples at 44.1 kHz) using overlap-add FFT costs roughly 5–15 ms of CPU time per sound source, per frame. A scene with 20 sound sources would immediately exceed the entire audio thread budget.
- **Both are completely unnecessary** for static geometry. The walls do not move. The room does not change. The simulation only needs to run once.

---

## The baking strategy

`nebula-audio` runs the expensive simulation **offline at bake time** and stores the results as static assets. At runtime the engine loads those assets and plays back pre-baked acoustic data instead of simulating anything.

There are two levels of fidelity in the output:

### Level 1 — Full Room Impulse Responses

A per-listener-position, per-frequency-band time-domain impulse response. This is the game-audio equivalent of the studio IR — it contains the complete acoustic fingerprint of the scene at that specific location, split across 8 octave bands (62.5 Hz through 8 kHz).

At runtime the engine:
1. Determines which pre-baked listener position is nearest to the camera.
2. Loads that listener's multi-band RIR.
3. Convolves the dry audio stream with the RIR using an overlap-add convolver.
4. Blends between the two nearest RIRs as the player moves.

The result is physically accurate, per-band reverb that changes smoothly as the player walks through the space — with zero ray tracing at runtime.

### Level 2 — Reverb Zone Parameters

Full convolution is still moderately expensive if you have many simultaneous sound sources. For those cases, `nebula-audio` also bakes a set of **reverb zone parameters** — scalar metrics derived from the RIRs that can drive a cheap algorithmic reverb (like the standard Schroeder or FDN reverb found in all game engines):

| Parameter | What it means to an audio engineer | Typical range |
|---|---|---|
| `t60` (RT60) | The time for the reverb tail to decay by 60 dB. The classic "size of room" parameter. | 0.2 s (bathroom) – 8 s (cathedral) |
| `edt` | Early Decay Time — RT60 measured only from the first 10 dB of decay. More perceptually relevant than full T60. | 0.1 – 2 s |
| `c80` | Clarity — ratio of early energy (< 80 ms) to late energy (dB). High C80 = clear and intelligible; low = muddy. | −5 to +15 dB |
| `d50` | Definition — fraction of total energy arriving within the first 50 ms. | 0.0 – 1.0 |
| `room_gain_db` | Perceived loudness boost from early reflections (the "room effect" a live room gives a singer). | 0 – 6 dB |
| `drr_db` | Direct-to-Reverb Ratio — how much louder the direct sound is than the reverb tail (controls apparent source distance). | −10 to +15 dB |
| `absorption[8]` | Per-band absorption estimate — how much each octave band decays relative to broadband. Drives EQ on the reverb tail. | 0.0 – 1.0 per band |

Think of these as the knobs on your reverb plugin — pre-dialled by physics simulation.

---

## How the simulation works

The baker combines two complementary acoustic modelling techniques:

### 1. Image-source method (early reflections)

For low-order reflections (typically orders 1–3), the baker uses the **image-source method**: it mirrors the listener position through each scene polygon to find the geometric location of the "virtual source" that would produce a first- or second-order specular reflection. It then tests direct visibility from the real source to this mirrored image. If visible, the reflection path is valid and its contribution — including travel-time delay, air absorption, and material absorption — is added to the impulse response.

Early reflections are perceptually critical. They arrive within the first 50–80 ms and define the spatial character of the room: whether it sounds wide or narrow, near or far, live or dead. Getting them right matters far more than having a long, accurate reverb tail.

### 2. Stochastic ray tracing (late reverb)

For the late reverb tail, the baker fires thousands of diffuse Monte Carlo rays from each listener position. Each ray bounces around the scene, accumulating energy decay across all 8 frequency bands according to:

- **Material absorption** — derived from the mesh's `audio_absorption` coefficient in `MaterialDesc`. Carpet absorbs high frequencies strongly; glass and concrete reflect them. This is the acoustic equivalent of the surface roughness used by the light baker.
- **Geometric spreading** — energy falls off with the square of distance as the wavefront expands.
- **Air absorption** — per ISO 9613-1, high frequencies (above ~4 kHz) are absorbed by air over long distances. This is why a cathedral sounds darker the longer the tail.

The energy decay envelopes are processed via **Schroeder backward integration** to extract the T60 per band — the slope of the energy decay curve.

---

## Frequency bands

The simulation uses 8 octave bands centred at:

```
62.5 Hz | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz | 8 kHz
```

If you have done room EQ or acoustic treatment work, these are the same bands you work with on a measurement microphone and REW/Room EQ Wizard. Bass frequencies (62.5–250 Hz) build up in room modes and decay slowly. Mid frequencies (500 Hz – 2 kHz) are the most perceptually important for speech intelligibility. High frequencies (4–8 kHz) are absorbed quickly by soft surfaces and air.

The per-band output means the reverb tail can be spectrally shaped correctly — a stone room has bright, long highs; a recording studio has dead, short highs. Without per-band simulation every room sounds the same regardless of material.

---

## Configuration

```rust
let config = AcousticConfig {
    listener_points: vec![
        ListenerPoint { position: [0.0, 1.5, 0.0], label: Some("entrance".into()) },
        ListenerPoint { position: [5.0, 1.5, 3.0], label: Some("centre".into()) },
    ],
    max_order:             2,     // Image-source reflection order (1–5)
    diffuse_rays:          512,   // Stochastic rays per listener (quality vs. speed)
    max_duration_secs:     2.0,   // Length of the baked IR (seconds)
    time_resolution_secs:  1.0 / 44100.0, // 44.1 kHz sample rate
    air_absorption:        [0.0002, 0.0004, 0.0006, 0.001, 0.002, 0.004, 0.008, 0.016],
    emit_reverb_zone:      true,
    occlusion_cell_size:   0.5,
};
```

**Presets:**

| Preset | Order | Rays | Duration | Use case |
|---|---|---|---|---|
| `fast()` | 1 | 64 | 0.5 s | Quick in-editor preview |
| `default()` | 2 | 512 | 2.0 s | Standard production |
| `ultra()` | 5 | 8192 | 5.0 s | High-accuracy film/AAA quality |

---

## GPU acceleration

Both stages (image-source and stochastic ray tracing) run on the GPU via WGSL compute shaders. The stochastic stage dispatches one workgroup per 64-ray bundle, so `diffuse_rays = 512` dispatches 8 workgroups simultaneously. This gives a ~40× speed-up over equivalent single-threaded CPU code.

For reference, simulating 512 diffuse rays with 5-bounce paths at 44.1 kHz across 8 frequency bands on CPU takes several seconds per listener point. The GPU completes the same work in under a second.

---

## How baking improves performance — the audio version

| Scenario | CPU cost | Quality |
|---|---|---|
| Real-time ray-traced acoustics | ~50–200 ms per frame | Perfect, dynamic |
| Runtime convolution with baked IR | ~2–5 ms per audio frame per source | Excellent, static geometry |
| Algorithmic reverb driven by baked parameters | ~0.01–0.1 ms per source | Good, static geometry |
| No reverb | 0 ms | Poor |

Baking moves the cost from **every frame** to **once at content-creation time**, with no perceptual difference for static geometry. The player cannot tell whether the cathedral reverb was simulated in real time or baked — the waveform coming out of the speakers is the same.

---

## Output

```rust
pub struct ImpulseResponse {
    pub listener_position:    [f32; 3],
    pub sample_rate:          u32,             // e.g. 44100
    pub bands:                [Vec<f32>; 8],   // Per-band time-domain samples
    pub t60_per_band:         [f32; 8],        // RT60 per octave band (seconds)
    pub broadband_t60:        f32,
    pub early_late_split_secs: f32,            // ~10% of broadband T60
}

pub struct ReverbZone {
    pub aabb_min:    [f32; 3],
    pub aabb_max:    [f32; 3],
    pub t60:         f32,
    pub edt:         f32,
    pub c80:         f32,
    pub d50:         f32,
    pub room_gain_db: f32,
    pub drr_db:      f32,
    pub absorption:  [f32; 8],  // Per-band absorption estimate
}
```

---

## License

MIT OR Apache-2.0
