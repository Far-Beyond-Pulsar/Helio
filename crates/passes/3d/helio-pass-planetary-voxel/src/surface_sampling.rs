use crate::{
    EXTRACTION_SAMPLE_COUNT, PlanetaryVoxelResidency, TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT,
    TRANSITION_FACE_SAMPLE_EDGE, transition_face_integer_basis,
};
use bytemuck::{Pod, Zeroable};
use helio_planet_voxel_core::{
    AddressError, GpuPageMeta, PAGE_EDGE, PageKey, PlanetId, PlanetPageKey, SourceGeneration,
    TRANSITION_FACE_MASK, TransitionFace,
};
use std::collections::BTreeSet;

const GATHER_WORKGROUP_SIZE: u32 = 64;
const INDIRECT_COMMAND_BYTES: u64 = 12;
const INDIRECT_COMMAND_COUNT: u32 = 8;

pub(crate) const REGULAR_EXTRACTION_INDIRECT_OFFSETS: [u64; 4] = [0, 12, 24, 36];
pub(crate) const TRANSITION_EXTRACTION_INDIRECT_OFFSETS: [u64; 4] = [48, 60, 72, 84];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetarySurfaceRequest {
    pub key: PlanetPageKey,
    pub generation: SourceGeneration,
    pub transition_mask: u8,
    pub dirty_microbricks: u64,
}

impl PlanetarySurfaceRequest {
    pub fn validate(self) -> Result<(), SurfaceSamplingError> {
        self.key.validate()?;
        if self.transition_mask & !TRANSITION_FACE_MASK != 0 {
            return Err(SurfaceSamplingError::TransitionMask(self.transition_mask));
        }
        if self.key.page.lod == 0 && self.transition_mask != 0 {
            return Err(SurfaceSamplingError::FinestLodTransition);
        }
        Ok(())
    }

    /// Exact page dependencies for the regular 34^3 halo and every enabled
    /// fine-side 67x67x3 transition slab. This returns page identities only;
    /// no renderer-specific scalar arrays are constructed on the CPU.
    pub fn required_pages(self) -> Result<BTreeSet<PlanetPageKey>, SurfaceSamplingError> {
        self.validate()?;
        let mut pages = BTreeSet::new();
        let page = self.key.page;
        let page_min = page.lod0_cell_min()?;
        let coarse_scale = 1_i64
            .checked_shl(u32::from(page.lod))
            .ok_or(AddressError::CoordinateOverflow)?;
        let regular_min = page_min.map(|value| value - coarse_scale);
        let regular_max = page_min.map(|value| value + PAGE_EDGE as i64 * coarse_scale);
        insert_page_box(
            self.key.planet,
            page.lod,
            regular_min,
            regular_max,
            &mut pages,
        )?;

        if self.transition_mask == 0 {
            return Ok(pages);
        }
        let fine_lod = page.lod - 1;
        let fine_scale = coarse_scale / 2;
        let page_span = PAGE_EDGE as i64 * coarse_scale;
        for face in TransitionFace::ALL {
            if self.transition_mask & face.bit() == 0 {
                continue;
            }
            let basis = transition_face_integer_basis(face);
            let mut minimum = [i64::MAX; 3];
            let mut maximum = [i64::MIN; 3];
            for u in [-1_i64, TRANSITION_FACE_SAMPLE_EDGE as i64] {
                for v in [-1_i64, TRANSITION_FACE_SAMPLE_EDGE as i64] {
                    for outward in [-1_i64, 1] {
                        let mut position = page_min;
                        for axis in 0..3 {
                            position[axis] = position[axis]
                                .checked_add(i64::from(basis.origin[axis]) * page_span)
                                .and_then(|value| {
                                    value
                                        .checked_add(i64::from(basis.u_axis[axis]) * u * fine_scale)
                                })
                                .and_then(|value| {
                                    value
                                        .checked_add(i64::from(basis.v_axis[axis]) * v * fine_scale)
                                })
                                .and_then(|value| {
                                    value.checked_add(
                                        i64::from(basis.outward[axis]) * outward * fine_scale,
                                    )
                                })
                                .ok_or(AddressError::CoordinateOverflow)?;
                            minimum[axis] = minimum[axis].min(position[axis]);
                            maximum[axis] = maximum[axis].max(position[axis]);
                        }
                    }
                }
            }
            insert_page_box(self.key.planet, fine_lod, minimum, maximum, &mut pages)?;
        }
        Ok(pages)
    }
}

fn insert_page_box(
    planet: PlanetId,
    lod: u8,
    minimum: [i64; 3],
    maximum: [i64; 3],
    output: &mut BTreeSet<PlanetPageKey>,
) -> Result<(), AddressError> {
    let minimum_page = PageKey::address_lod0_cell(lod, minimum)?.0.page_xyz;
    let maximum_page = PageKey::address_lod0_cell(lod, maximum)?.0.page_xyz;
    for z in minimum_page[2]..=maximum_page[2] {
        for y in minimum_page[1]..=maximum_page[1] {
            for x in minimum_page[0]..=maximum_page[0] {
                output.insert(PlanetPageKey::new(planet, PageKey::new(lod, [x, y, z])));
            }
        }
    }
    Ok(())
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSurfaceGatherJob {
    pub planet_id: [u32; 4],
    pub relative_lod0_cell_min: [i32; 3],
    pub lod: u32,
    pub generation_low: u32,
    pub generation_high: u32,
    pub transition_mask: u32,
    pub target_slot: u32,
    pub residency_epoch_low: u32,
    pub residency_epoch_high: u32,
    pub _pad: [u32; 2],
}

impl GpuSurfaceGatherJob {
    pub fn new(
        request: PlanetarySurfaceRequest,
        metadata: GpuPageMeta,
        residency_epoch: u64,
    ) -> Self {
        let mut planet_id = [0_u32; 4];
        for (word, bytes) in planet_id
            .iter_mut()
            .zip(request.key.planet.0.chunks_exact(4))
        {
            *word = u32::from_le_bytes(bytes.try_into().expect("planet words are four bytes"));
        }
        Self {
            planet_id,
            relative_lod0_cell_min: metadata.relative_lod0_cell_min,
            lod: metadata.lod,
            generation_low: metadata.generation_low,
            generation_high: metadata.generation_high,
            transition_mask: u32::from(request.transition_mask),
            target_slot: metadata.slot,
            residency_epoch_low: residency_epoch as u32,
            residency_epoch_high: (residency_epoch >> 32) as u32,
            _pad: [0; 2],
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSurfaceGatherCounters {
    pub regular_samples: u32,
    pub transition_samples: u32,
    pub table_probes: u32,
    pub page_misses: u32,
    pub stale_targets: u32,
    pub completed: u32,
    pub _pad: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceSamplerResourceStats {
    pub buffers: u32,
    pub allocated_bytes: u64,
}

pub struct GpuSurfaceSampler {
    regular_pipeline: wgpu::ComputePipeline,
    transition_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    job_buffer: wgpu::Buffer,
    counters_buffer: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    resource_stats: SurfaceSamplerResourceStats,
}

impl GpuSurfaceSampler {
    pub(crate) fn new(
        device: &wgpu::Device,
        residency: &PlanetaryVoxelResidency,
        regular_samples: &wgpu::Buffer,
        transition_samples: &wgpu::Buffer,
    ) -> Result<Self, SurfaceSamplingError> {
        validate_limits(&device.limits())?;
        let job_buffer = create_buffer(
            device,
            "Planetary Surface Gather Job",
            core::mem::size_of::<GpuSurfaceGatherJob>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let counters_buffer = create_buffer(
            device,
            "Planetary Surface Gather Counters",
            core::mem::size_of::<GpuSurfaceGatherCounters>() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let indirect_buffer = create_buffer(
            device,
            "Planetary Surface Extraction Indirect Dispatches",
            u64::from(INDIRECT_COMMAND_COUNT) * INDIRECT_COMMAND_BYTES,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planetary Surface Gather Bind Group Layout"),
            entries: &[
                uniform_layout_entry(0),
                uniform_layout_entry(1),
                storage_layout_entry(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_layout_entry(4, false),
                storage_layout_entry(5, false),
                storage_layout_entry(6, false),
                storage_layout_entry(7, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planetary Surface Gather Pipeline Layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Surface Gather Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::SURFACE_GATHER_WGSL.into()),
        });
        let create_pipeline = |label: &'static str, entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let regular_pipeline =
            create_pipeline("Planetary Regular Surface Gather", "gather_regular");
        let transition_pipeline =
            create_pipeline("Planetary Transition Surface Gather", "gather_transition");
        let finalize_pipeline =
            create_pipeline("Planetary Surface Gather Finalize", "finalize_gather");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Surface Gather Bind Group"),
            layout: &layout,
            entries: &[
                buffer_entry(0, &job_buffer),
                buffer_entry(1, residency.residency_uniform_buffer()),
                buffer_entry(2, residency.page_table_buffer()),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(residency.atlas_view()),
                },
                buffer_entry(4, regular_samples),
                buffer_entry(5, transition_samples),
                buffer_entry(6, &counters_buffer),
                buffer_entry(7, &indirect_buffer),
            ],
        });
        Ok(Self {
            regular_pipeline,
            transition_pipeline,
            finalize_pipeline,
            bind_group,
            job_buffer,
            counters_buffer,
            indirect_buffer,
            resource_stats: SurfaceSamplerResourceStats {
                buffers: 3,
                allocated_bytes: core::mem::size_of::<GpuSurfaceGatherJob>() as u64
                    + core::mem::size_of::<GpuSurfaceGatherCounters>() as u64
                    + u64::from(INDIRECT_COMMAND_COUNT) * INDIRECT_COMMAND_BYTES,
            },
        })
    }

    pub(crate) fn prepare(&self, queue: &wgpu::Queue, job: GpuSurfaceGatherJob) {
        queue.write_buffer(&self.job_buffer, 0, bytemuck::bytes_of(&job));
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.counters_buffer, 0, None);
        encoder.clear_buffer(&self.indirect_buffer, 0, None);
        dispatch(
            encoder,
            &self.regular_pipeline,
            &self.bind_group,
            (EXTRACTION_SAMPLE_COUNT as u32).div_ceil(GATHER_WORKGROUP_SIZE),
            "Planetary Regular Samples Gather",
        );
        dispatch(
            encoder,
            &self.transition_pipeline,
            &self.bind_group,
            (TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT as u32).div_ceil(GATHER_WORKGROUP_SIZE),
            "Planetary Transition Samples Gather",
        );
        dispatch(
            encoder,
            &self.finalize_pipeline,
            &self.bind_group,
            1,
            "Planetary Surface Gather Finalize",
        );
    }

    pub(crate) fn indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_buffer
    }

    pub fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.counters_buffer
    }

    pub const fn resource_stats(&self) -> SurfaceSamplerResourceStats {
        self.resource_stats
    }
}

fn validate_limits(limits: &wgpu::Limits) -> Result<(), SurfaceSamplingError> {
    if limits.max_storage_buffers_per_shader_stage < 5 {
        return Err(SurfaceSamplingError::DeviceLimit {
            name: "storage buffers per shader stage",
            required: 5,
            available: u64::from(limits.max_storage_buffers_per_shader_stage),
        });
    }
    if limits.max_uniform_buffers_per_shader_stage < 2 {
        return Err(SurfaceSamplingError::DeviceLimit {
            name: "uniform buffers per shader stage",
            required: 2,
            available: u64::from(limits.max_uniform_buffers_per_shader_stage),
        });
    }
    if limits.max_sampled_textures_per_shader_stage < 1 {
        return Err(SurfaceSamplingError::DeviceLimit {
            name: "sampled textures per shader stage",
            required: 1,
            available: u64::from(limits.max_sampled_textures_per_shader_stage),
        });
    }
    let workgroup_width = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x);
    if workgroup_width < GATHER_WORKGROUP_SIZE {
        return Err(SurfaceSamplingError::DeviceLimit {
            name: "compute workgroup width",
            required: u64::from(GATHER_WORKGROUP_SIZE),
            available: u64::from(workgroup_width),
        });
    }
    Ok(())
}

fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
    label: &'static str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceSamplingError {
    #[error(transparent)]
    Address(#[from] AddressError),
    #[error("transition mask {0:#010b} uses bits outside the six page faces")]
    TransitionMask(u8),
    #[error("LOD0 cannot own a transition surface")]
    FinestLodTransition,
    #[error("surface sampler needs {required} {name}; device provides {available}")]
    DeviceLimit {
        name: &'static str,
        required: u64,
        available: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_planet_voxel_core::{
        CellWord, PAGE_CELL_COUNT, PageUpload, PlanetFrameUniform, PlanetPosition,
    };
    use std::sync::mpsc;

    #[test]
    fn regular_dependencies_are_the_exact_twenty_seven_neighbor_pages() {
        let request = PlanetarySurfaceRequest {
            key: PlanetPageKey::new(PlanetId([3; 16]), PageKey::new(4, [-2, 5, -7])),
            generation: SourceGeneration::default(),
            transition_mask: 0,
            dirty_microbricks: u64::MAX,
        };
        let pages = request.required_pages().unwrap();
        assert_eq!(pages.len(), 27);
        assert!(pages.contains(&request.key));
        assert!(pages.contains(&PlanetPageKey::new(
            request.key.planet,
            PageKey::new(4, [-3, 4, -8]),
        )));
        assert!(pages.contains(&PlanetPageKey::new(
            request.key.planet,
            PageKey::new(4, [-1, 6, -6]),
        )));
    }

    #[test]
    fn transition_dependencies_use_fine_pages_on_signed_boundaries() {
        let request = PlanetarySurfaceRequest {
            key: PlanetPageKey::new(PlanetId([9; 16]), PageKey::new(3, [-1, 0, -1])),
            generation: SourceGeneration::default(),
            transition_mask: TransitionFace::NegativeX.bit() | TransitionFace::PositiveZ.bit(),
            dirty_microbricks: u64::MAX,
        };
        let pages = request.required_pages().unwrap();
        assert!(pages.iter().any(|key| key.page.lod == 2));
        assert!(pages.iter().any(|key| key.page.page_xyz[0] < 0));
        assert!(pages.iter().any(|key| key.page.page_xyz[2] >= 0));
    }

    #[test]
    fn lod0_rejects_transition_ownership() {
        let request = PlanetarySurfaceRequest {
            key: PlanetPageKey::default(),
            generation: SourceGeneration::default(),
            transition_mask: 1,
            dirty_microbricks: u64::MAX,
        };
        assert_eq!(
            request.validate(),
            Err(SurfaceSamplingError::FinestLodTransition)
        );
    }

    #[test]
    fn compact_request_replaces_the_expanded_cpu_sample_payload() {
        let expanded_sample_bytes = (EXTRACTION_SAMPLE_COUNT
            + TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT)
            * core::mem::size_of::<CellWord>();
        assert_eq!(expanded_sample_bytes, 480_424);
        assert_eq!(core::mem::size_of::<PlanetarySurfaceRequest>(), 80);
        assert!(
            expanded_sample_bytes
                >= core::mem::size_of::<PlanetarySurfaceRequest>().saturating_mul(6_000)
        );
    }

    #[test]
    fn horizon_plan_sampling_support_stays_planet_size_independent_and_bounded() {
        for minimum_lod in [0, 2, 5] {
            let plan = crate::HorizonLodFixturePlan::build_with_minimum_lod(
                [63_710_000, -1, -17],
                11,
                minimum_lod,
                192,
            )
            .unwrap();
            let mut dependencies = BTreeSet::new();
            let mut maximum_request_dependencies = 0;
            for page in plan.topology().pages() {
                let request = PlanetarySurfaceRequest {
                    key: PlanetPageKey::new(PlanetId([7; 16]), page),
                    generation: SourceGeneration::new(1, 1),
                    transition_mask: plan.topology().transition_mask(page).unwrap(),
                    dirty_microbricks: u64::MAX,
                };
                let request_dependencies = request.required_pages().unwrap();
                maximum_request_dependencies =
                    maximum_request_dependencies.max(request_dependencies.len());
                dependencies.extend(request_dependencies);
            }
            eprintln!(
                "PLANETARY_SAMPLING_SUPPORT: minimum_lod={minimum_lod} visible={} union={} max_request={maximum_request_dependencies}",
                plan.topology().stats().pages,
                dependencies.len()
            );
            assert!(maximum_request_dependencies <= 96);
        }
    }

    #[test]
    fn gpu_gather_matches_canonical_pages_across_signed_regular_and_transition_boundaries() {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let Some(adapter) = request_test_adapter(&instance).await else {
                eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: surface gather");
                return;
            };
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Planetary Surface Gather Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .expect("surface gather adapter must create a device");
            device.on_uncaptured_error(std::sync::Arc::new(|error| {
                panic!("planetary surface gather GPU validation error: {error:?}");
            }));

            let planet = PlanetId([0x5a; 16]);
            let key = PlanetPageKey::new(planet, PageKey::new(2, [-1, -2, 1]));
            let request = PlanetarySurfaceRequest {
                key,
                generation: SourceGeneration::new(1, 1),
                transition_mask: 0x3f,
                dirty_microbricks: u64::MAX,
            };
            let dependencies = request.required_pages().unwrap();
            let config = crate::PlanetaryVoxelGpuConfig::new(
                dependencies.len() as u32,
                256,
                64,
                dependencies.len() as u32,
                256,
            )
            .unwrap();
            let mut residency = PlanetaryVoxelResidency::new(&device, &queue, config).unwrap();
            let camera = PlanetPosition::from_lod0_cell([32, 0, 0]);
            let frame = PlanetFrameUniform::from_camera(planet, camera, 1);
            assert_eq!(frame.frame_origin_lod0_cell(), [32, 0, 0]);
            assert_ne!(
                key.page
                    .relative_lod0_cell_min(frame.frame_origin_lod0_cell())
                    .unwrap()[0]
                    .rem_euclid(key.page.lod0_cell_span().unwrap() as i32),
                0,
                "the regression requires an LOD0-snapped frame that is not coarse-page aligned"
            );
            residency.set_planet_frame(&queue, frame).unwrap();

            let mut uploads = Vec::with_capacity(dependencies.len());
            for dependency in dependencies {
                let generation = if dependency == key {
                    request.generation
                } else {
                    SourceGeneration::new(1, uploads.len() as u64 + 2)
                };
                uploads.push(
                    PageUpload::new(
                        dependency,
                        generation,
                        canonical_page_cells(dependency.page),
                    )
                    .unwrap(),
                );
            }
            residency
                .apply_upload_batch(&device, &queue, uploads)
                .unwrap();

            let regular_samples = test_storage_buffer(
                &device,
                "Planetary Gather Regular Test Samples",
                EXTRACTION_SAMPLE_COUNT,
            );
            let transition_samples = test_storage_buffer(
                &device,
                "Planetary Gather Transition Test Samples",
                TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT,
            );
            let sampler =
                GpuSurfaceSampler::new(&device, &residency, &regular_samples, &transition_samples)
                    .unwrap();
            let resident = residency.cache().resident(key).unwrap();
            let metadata = GpuPageMeta::new(
                key.page,
                frame.frame_origin_lod0_cell(),
                resident.slot,
                resident.publication_generation,
                request.transition_mask,
            )
            .unwrap();
            sampler.prepare(
                &queue,
                GpuSurfaceGatherJob::new(request, metadata, residency.publication_epoch()),
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Planetary Surface Gather Test Encoder"),
            });
            sampler.encode(&mut encoder);
            queue.submit([encoder.finish()]);

            let regular =
                read_buffer::<CellWord>(&device, &queue, &regular_samples, EXTRACTION_SAMPLE_COUNT);
            let page_min = key.page.lod0_cell_min().unwrap();
            let scale = 1_i64 << key.page.lod;
            for z in 0..34_i64 {
                for y in 0..34_i64 {
                    for x in 0..34_i64 {
                        let index = (x + y * 34 + z * 34 * 34) as usize;
                        let position = [
                            page_min[0] + (x - 1) * scale,
                            page_min[1] + (y - 1) * scale,
                            page_min[2] + (z - 1) * scale,
                        ];
                        assert_eq!(regular[index], canonical_cell(position));
                    }
                }
            }

            let transition = read_buffer::<CellWord>(
                &device,
                &queue,
                &transition_samples,
                TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT,
            );
            let fine_scale = scale / 2;
            let coarse_span = PAGE_EDGE as i64 * scale;
            for face in TransitionFace::ALL {
                let basis = transition_face_integer_basis(face);
                let face_offset = usize::from(face.index()) * 67 * 67 * 3;
                for layer in 0..3_i64 {
                    for v in 0..67_i64 {
                        for u in 0..67_i64 {
                            let mut position = page_min;
                            for (axis, coordinate) in position.iter_mut().enumerate() {
                                *coordinate += i64::from(basis.origin[axis]) * coarse_span
                                    + i64::from(basis.u_axis[axis]) * (u - 1) * fine_scale
                                    + i64::from(basis.v_axis[axis]) * (v - 1) * fine_scale
                                    + i64::from(basis.outward[axis]) * (layer - 1) * fine_scale;
                            }
                            let index = face_offset
                                + (layer as usize * 67 * 67)
                                + (v as usize * 67)
                                + u as usize;
                            assert_eq!(transition[index], canonical_cell(position));
                        }
                    }
                }
            }

            let counters = read_buffer::<GpuSurfaceGatherCounters>(
                &device,
                &queue,
                sampler.counters_buffer(),
                1,
            );
            assert_eq!(counters[0].regular_samples, EXTRACTION_SAMPLE_COUNT as u32);
            assert_eq!(
                counters[0].transition_samples,
                TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT as u32
            );
            assert_eq!(counters[0].page_misses, 0);
            assert_eq!(counters[0].stale_targets, 0);
            assert_eq!(counters[0].completed, 1);

            let commands = read_buffer::<u32>(
                &device,
                &queue,
                sampler.indirect_buffer(),
                INDIRECT_COMMAND_COUNT as usize * 3,
            );
            assert_eq!(
                commands,
                vec![
                    512, 1, 1, 128, 1, 1, 1, 1, 1, 512, 1, 1, 96, 1, 1, 24, 1, 1, 1, 1, 1, 96, 1,
                    1,
                ]
            );
        });
    }

    fn canonical_page_cells(page: PageKey) -> Vec<CellWord> {
        let minimum = page.lod0_cell_min().unwrap();
        let scale = 1_i64 << page.lod;
        let mut cells = Vec::with_capacity(PAGE_CELL_COUNT);
        for z in 0..PAGE_EDGE as i64 {
            for y in 0..PAGE_EDGE as i64 {
                for x in 0..PAGE_EDGE as i64 {
                    cells.push(canonical_cell([
                        minimum[0] + x * scale,
                        minimum[1] + y * scale,
                        minimum[2] + z * scale,
                    ]));
                }
            }
        }
        cells
    }

    fn canonical_cell(position: [i64; 3]) -> CellWord {
        let mixed = position[0]
            .wrapping_mul(17)
            .wrapping_add(position[1].wrapping_mul(31))
            .wrapping_add(position[2].wrapping_mul(43));
        let density = (mixed.rem_euclid(30_001) - 15_000) as i16;
        let material = (mixed.wrapping_mul(7).rem_euclid(255) + 1) as u8;
        let flags = mixed.wrapping_mul(11).rem_euclid(64) as u8;
        CellWord::new(density, material, flags)
    }

    fn test_storage_buffer(
        device: &wgpu::Device,
        label: &'static str,
        element_count: usize,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (element_count * core::mem::size_of::<CellWord>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn read_buffer<T: Pod + Copy>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        count: usize,
    ) -> Vec<T> {
        let size = (count * core::mem::size_of::<T>()) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Surface Gather Test Readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Planetary Surface Gather Test Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(source, 0, &readback, 0, size);
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("surface gather readback callback must run")
            .expect("surface gather readback mapping must succeed");
        let mapped = slice
            .get_mapped_range()
            .expect("surface gather readback range must be available");
        let values = bytemuck::cast_slice::<u8, T>(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        values
    }

    async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
        for force_fallback_adapter in [false, true] {
            if let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter,
                    apply_limit_buckets: false,
                })
                .await
            {
                return Some(adapter);
            }
        }
        None
    }
}
