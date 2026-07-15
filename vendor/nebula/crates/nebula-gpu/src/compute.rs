use std::borrow::Cow;

/// A compiled compute pipeline with its bind group layouts.
pub struct ComputePipeline {
    pub pipeline:       wgpu::ComputePipeline,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

impl ComputePipeline {
    /// Build a compute pipeline from inline WGSL source.
    pub fn from_wgsl(
        device:     &wgpu::Device,
        label:      &str,
        wgsl:       &str,
        entry:      &str,
        layouts:    &[&wgpu::BindGroupLayout],
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some(&format!("{label}_shader")),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
        });

        let opt_layouts: Vec<Option<&wgpu::BindGroupLayout>> =
            layouts.iter().map(|&l| Some(l)).collect();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:          Some(&format!("{label}_layout")),
            bind_group_layouts:   &opt_layouts,
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some(label),
            layout:      Some(&pipeline_layout),
            module:      &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layouts: Vec::new(), // caller holds layouts externally
        }
    }
}

// ── ConvenienceComputePass ─────────────────────────────────────────────────────

/// Builder wrapper for dispatching a compute pass.
pub struct ComputePass<'enc> {
    inner: wgpu::ComputePass<'enc>,
}

impl<'enc> ComputePass<'enc> {
    pub fn new(encoder: &'enc mut wgpu::CommandEncoder, label: &str) -> Self {
        Self {
            inner: encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            }),
        }
    }

    pub fn set_pipeline(&mut self, pipeline: &'enc ComputePipeline) {
        self.inner.set_pipeline(&pipeline.pipeline);
    }

    pub fn set_bind_group(&mut self, index: u32, group: &'enc wgpu::BindGroup) {
        self.inner.set_bind_group(index, group, &[]);
    }

    pub fn dispatch_workgroups(mut self, x: u32, y: u32, z: u32) {
        self.inner.dispatch_workgroups(x, y, z);
    }
}
