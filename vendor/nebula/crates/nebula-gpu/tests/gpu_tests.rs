use nebula_gpu::{MAX_TEXTURE_DIM, WORKGROUP_SIZE};

// ── Compile-time constants ────────────────────────────────────────────────────

#[test]
fn max_texture_dim_is_8192() {
    assert_eq!(MAX_TEXTURE_DIM, 8192);
}

#[test]
fn workgroup_size_is_8() {
    assert_eq!(WORKGROUP_SIZE, 8);
}

#[test]
fn workgroup_size_squared_fits_in_u32() {
    let total = WORKGROUP_SIZE * WORKGROUP_SIZE;
    assert!(total <= 1024, "workgroup total ({total}) should not exceed typical GPU limit of 1024");
}

#[test]
fn max_texture_dim_is_power_of_two() {
    assert!(MAX_TEXTURE_DIM.is_power_of_two());
}

#[test]
fn workgroup_size_is_power_of_two() {
    assert!(WORKGROUP_SIZE.is_power_of_two());
}

// ── TextureFormat2D constants ─────────────────────────────────────────────────

#[test]
fn texture_format_r32f_is_distinct_from_rgba32f() {
    use nebula_gpu::TextureFormat2D;
    // These are wgpu::TextureFormat constants; verifying they are different
    // values demonstrates the aliases are set correctly.
    assert_ne!(TextureFormat2D::R32F, TextureFormat2D::RGBA32F);
}

#[test]
fn texture_format_rgba16f_is_distinct_from_rgba32f() {
    use nebula_gpu::TextureFormat2D;
    assert_ne!(TextureFormat2D::RGBA16F, TextureFormat2D::RGBA32F);
}

#[test]
fn texture_format_rgba8_is_distinct_from_r32f() {
    use nebula_gpu::TextureFormat2D;
    assert_ne!(TextureFormat2D::RGBA8, TextureFormat2D::R32F);
}

#[test]
fn texture_format_rg32f_is_distinct_from_r32f() {
    use nebula_gpu::TextureFormat2D;
    assert_ne!(TextureFormat2D::RG32F, TextureFormat2D::R32F);
}

#[test]
fn texture_format_all_five_constants_are_distinct() {
    use nebula_gpu::TextureFormat2D;
    let formats = [
        TextureFormat2D::RGBA8,
        TextureFormat2D::RGBA16F,
        TextureFormat2D::RGBA32F,
        TextureFormat2D::R32F,
        TextureFormat2D::RG32F,
    ];
    // Every pair should be distinct.
    for i in 0..formats.len() {
        for j in (i + 1)..formats.len() {
            assert_ne!(formats[i], formats[j], "formats[{i}] == formats[{j}]");
        }
    }
}

// ── Type existence (compile-time checks) ──────────────────────────────────────

#[test]
fn storage_buffer_type_is_importable() {
    // Holds a phantom Option so the generic is known.
    let _: Option<nebula_gpu::StorageBuffer<u32>> = None;
}

#[test]
fn uniform_buffer_type_is_importable() {
    let _: Option<nebula_gpu::UniformBuffer<[f32; 4]>> = None;
}

#[test]
fn bake_texture_type_is_importable() {
    let _: Option<&nebula_gpu::BakeTexture> = None;
}

#[test]
fn bake_texture_array_type_is_importable() {
    let _: Option<&nebula_gpu::BakeTextureArray> = None;
}

#[test]
fn gpu_readback_type_is_importable() {
    let _: Option<&nebula_gpu::GpuReadback> = None;
}

#[test]
fn compute_pipeline_type_is_importable() {
    let _: Option<&nebula_gpu::ComputePipeline> = None;
}

// ── GPU-adapter-gated tests ───────────────────────────────────────────────────

#[test]
#[ignore = "requires GPU adapter"]
fn storage_buffer_zeroed_does_not_panic() {
    use nebula_core::context::BakeContext;
    use nebula_gpu::StorageBuffer;

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let _buf = StorageBuffer::<u32>::zeroed(&ctx.device, "test_storage", 64, false);
    });
}

#[test]
#[ignore = "requires GPU adapter"]
fn storage_buffer_from_slice_does_not_panic() {
    use nebula_core::context::BakeContext;
    use nebula_gpu::StorageBuffer;

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let data = vec![1u32, 2, 3, 4];
        let buf = StorageBuffer::from_slice(&ctx.device, "test_storage", &data, true);
        assert_eq!(buf.len, 4);
    });
}

#[test]
#[ignore = "requires GPU adapter"]
fn uniform_buffer_creation_does_not_panic() {
    use nebula_core::context::BakeContext;
    use nebula_gpu::UniformBuffer;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct TestUniform { x: f32, y: f32, z: f32, w: f32 }

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let uniform = TestUniform { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let _buf = UniformBuffer::new(&ctx.device, "test_uniform", &uniform);
    });
}

#[test]
#[ignore = "requires GPU adapter"]
fn bake_texture_creation_does_not_panic() {
    use nebula_core::context::BakeContext;
    use nebula_gpu::{BakeTexture, TextureFormat2D};

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let _tex = BakeTexture::new(
            &ctx.device,
            "test_tex",
            64, 64,
            TextureFormat2D::R32F,
            1,
            wgpu::TextureUsages::empty(),
        );
    });
}

#[test]
#[ignore = "requires GPU adapter"]
fn bake_texture_dimensions_match_request() {
    use nebula_core::context::BakeContext;
    use nebula_gpu::{BakeTexture, TextureFormat2D};

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let tex = BakeTexture::new(
            &ctx.device,
            "test_tex_dims",
            128, 256,
            TextureFormat2D::RGBA32F,
            1,
            wgpu::TextureUsages::empty(),
        );
        assert_eq!(tex.width, 128);
        assert_eq!(tex.height, 256);
        assert_eq!(tex.mip_levels, 1);
    });
}
