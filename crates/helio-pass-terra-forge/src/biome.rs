use crate::gpu_types::GpuMaterial;

pub fn default_palette() -> Vec<GpuMaterial> {
    let mut palette = vec![
        GpuMaterial {
            color: [0.0, 0.0, 0.0],
            roughness: 1.0,
        };
        256
    ];
    palette[1] = GpuMaterial {
        color: [0.6, 0.55, 0.5],
        roughness: 0.9,
    };
    palette[2] = GpuMaterial {
        color: [0.3, 0.6, 0.2],
        roughness: 0.85,
    };
    palette[3] = GpuMaterial {
        color: [0.5, 0.35, 0.2],
        roughness: 0.95,
    };
    palette
}
