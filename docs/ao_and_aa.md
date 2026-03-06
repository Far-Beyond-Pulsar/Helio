# Ambient Occlusion and Anti-Aliasing

This document describes the AO (Ambient Occlusion) and AA (Anti-Aliasing) features available in Helio Render V2.

## Ambient Occlusion (SSAO)

Screen-Space Ambient Occlusion (SSAO) adds realistic shadowing in crevices and corners by calculating occlusion based on the depth buffer and normals.

### Enabling SSAO

```rust
use helio_render_v2::{Renderer, RendererConfig, passes::SsaoConfig};

let config = RendererConfig::new(width, height, surface_format, features)
    .with_ssao(); // Enable with default settings

// Or with custom settings:
let config = RendererConfig::new(width, height, surface_format, features)
    .with_ssao_config(SsaoConfig {
        radius: 0.5,      // Sampling radius in world space
        bias: 0.025,      // Depth bias to prevent self-occlusion
        power: 2.0,       // Contrast power (higher = darker)
        samples: 16,      // Number of samples (more = higher quality, slower)
    });
```

### SSAO Parameters

- **radius**: The world-space radius for sampling occlusion. Larger values create wider ambient shadows.
- **bias**: Depth bias to prevent self-shadowing artifacts. Increase if you see "acne" patterns.
- **power**: Contrast exponent applied to the final AO value. Higher values make occlusion darker.
- **samples**: Number of hemisphere samples. More samples = better quality but slower performance.

## Anti-Aliasing Modes

Helio supports 4 different anti-aliasing modes, each with different quality/performance trade-offs:

### 1. FXAA (Fast Approximate Anti-Aliasing)

Fast post-processing AA that smooths edges by detecting and blending high-contrast pixels.

**Pros**: Very fast, works on all hardware  
**Cons**: Can blur textures, not as high quality as other methods

```rust
use helio_render_v2::passes::AntiAliasingMode;

let config = RendererConfig::new(width, height, surface_format, features)
    .with_aa(AntiAliasingMode::Fxaa);
```

### 2. SMAA (Subpixel Morphological Anti-Aliasing)

Advanced post-processing AA that detects edge patterns and applies high-quality filtering.

**Pros**: Better quality than FXAA, still fast  
**Cons**: More complex, requires pattern lookup textures

```rust
let config = RendererConfig::new(width, height, surface_format, features)
    .with_aa(AntiAliasingMode::Smaa);
```

### 3. TAA (Temporal Anti-Aliasing)

Accumulates multiple jittered frames over time for very high quality AA.

**Pros**: Highest quality, removes sub-pixel flickering  
**Cons**: Can cause ghosting with fast motion, requires jittered camera projection

```rust
use helio_render_v2::passes::{AntiAliasingMode, TaaConfig};

let config = RendererConfig::new(width, height, surface_format, features)
    .with_aa(AntiAliasingMode::Taa)
    .with_taa_config(TaaConfig {
        feedback_min: 0.88,  // Minimum history blend factor
        feedback_max: 0.97,  // Maximum history blend factor
    });
```

**Note**: TAA requires camera jitter to be applied each frame. The TAA pass provides jitter offsets via `TaaPass::get_jitter_offset()`.

### 4. MSAA (Multisample Anti-Aliasing)

Hardware-based supersampling that renders at higher resolution per pixel.

**Pros**: High quality, no post-processing artifacts  
**Cons**: High memory usage, performance intensive

```rust
use helio_render_v2::passes::{AntiAliasingMode, MsaaSamples};

let config = RendererConfig::new(width, height, surface_format, features)
    .with_aa(AntiAliasingMode::Msaa(MsaaSamples::X4)); // 2x, 4x, or 8x

```

MSAA sample counts:
- `MsaaSamples::X2` - 2x sampling (good performance)
- `MsaaSamples::X4` - 4x sampling (balanced)
- `MsaaSamples::X8` - 8x sampling (high quality, expensive)

## Combining AO and AA

You can enable both AO and AA together:

```rust
use helio_render_v2::passes::{AntiAliasingMode, SsaoConfig};

let config = RendererConfig::new(width, height, surface_format, features)
    .with_ssao()
    .with_aa(AntiAliasingMode::Fxaa);
```

## Performance Tips

1. **SSAO**: Start with 16 samples and adjust based on performance needs. 8 samples is faster but noisier.
2. **FXAA**: Best for real-time applications where performance is critical.
3. **TAA**: Best for cinematic/high-quality applications. Ensure camera movement is smooth.
4. **MSAA**: Best when memory bandwidth is available. Combine with FXAA for even better quality.

## Example Scene

```rust
use helio_render_v2::{Renderer, RendererConfig, passes::{AntiAliasingMode, SsaoConfig}};

let config = RendererConfig::new(1280, 720, surface_format, features)
    .with_ssao_config(SsaoConfig {
        radius: 0.75,
        bias: 0.02,
        power: 2.5,
        samples: 32,
    })
    .with_aa(AntiAliasingMode::Taa);

let renderer = Renderer::new(device, queue, config)?;
```

This configuration enables high-quality SSAO with TAA for a cinematic look with minimal aliasing and realistic ambient shadows.
