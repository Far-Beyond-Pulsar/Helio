use crate::*;
use helio_core::GpuCameraUniforms;
use std::{sync::mpsc, sync::Arc};

#[test]
fn adjacent_scene_chunks_cull_shared_block_faces_and_draw_visible_pixels() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => {
                eprintln!("GPU draw test skipped: no adapter");
                return;
            }
        };
        if !adapter
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
        {
            eprintln!("GPU draw test skipped: adapter lacks INDIRECT_FIRST_INSTANCE");
            return;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Voxel Scene Draw Test"),
                required_features: wgpu::Features::INDIRECT_FIRST_INSTANCE,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("GPU device");
        device.on_uncaptured_error(Arc::new(|error| panic!("voxel draw validation: {error:?}")));
        let mut pass =
            VoxelMeshPass::new_composited(&device, &queue, wgpu::TextureFormat::Rgba8Unorm);
        let filled = [1u8; VOXEL_CHUNK_SAMPLES];
        let domain = VoxelDomain::Bounded {
            min: [0, 0, 0],
            max: [1, 0, 0],
            max_lod: 0,
        };
        let mut dirty = [DirtyBrick::zeroed(); 2];
        for slot in 0..2u32 {
            let key = VoxelChunkKey::new(i64::from(slot), 0, 0, 0);
            let words = bake_padded_chunk_with_policy(
                key,
                domain,
                VoxelMissingChunkPolicy::KnownAir,
                |neighbor| {
                    (neighbor == VoxelChunkKey::new(0, 0, 0, 0)
                        || neighbor == VoxelChunkKey::new(1, 0, 0, 0))
                    .then_some(&filled[..])
                },
            )
            .unwrap();
            queue.write_buffer(
                &pass.voxel_data_buf,
                u64::from(slot) * VOXEL_MESH_BRICK_VOXEL_WORDS * 4,
                bytemuck::cast_slice(&words),
            );
            let meta = GpuBrickMeta {
                data_offset: slot * VOXEL_MESH_BRICK_VOXEL_WORDS as u32,
                occupancy: 1,
            };
            queue.write_buffer(
                &pass.brick_meta_buf,
                u64::from(slot) * std::mem::size_of::<GpuBrickMeta>() as u64,
                bytemuck::bytes_of(&meta),
            );
            let mut map = [0u32; 256];
            map[0] = 1;
            map[1] = slot;
            queue.write_buffer(
                &pass.material_map_buf,
                u64::from(slot) * 256 * 4,
                bytemuck::cast_slice(&map),
            );
            dirty[slot as usize] = DirtyBrick {
                brick_slot: slot,
                volume_id: 0,
                mode: VOXEL_MODE_CUBES,
                _pad: 1,
                origin_size: [0.0, 0.0, 0.0, 0.1],
            };
        }
        queue.write_buffer(&pass.dirty_brick_buf, 0, bytemuck::cast_slice(&dirty));
        let mut origins = [[0.0f32; 4]; VOXEL_MESH_MAX_BRICKS as usize];
        origins[0] = [-0.8, -0.4, 0.0, 0.0];
        origins[1] = [0.0, -0.4, 0.0, 0.0];
        queue.write_buffer(&pass.render_origin_buf, 0, bytemuck::cast_slice(&origins));
        queue.write_buffer(
            &pass.meshlet_params_buf,
            0,
            bytemuck::bytes_of(&MeshletParams::zeroed()),
        );

        let camera = GpuCameraUniforms::new(
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
            glam::Vec3::ZERO,
            0.01,
            10.0,
            0,
            [0.0; 2],
            glam::Mat4::IDENTITY,
        );
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Test Camera"),
            contents: bytemuck::cast_slice(&[camera, camera]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut red = helio_mats::GpuMaterial::zeroed();
        red.base_color = [1.0, 0.0, 0.0, 1.0];
        let mut green = helio_mats::GpuMaterial::zeroed();
        green.base_color = [0.0, 1.0, 0.0, 1.0];
        let scene_materials_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Test SceneDB Materials"),
            contents: bytemuck::cast_slice(&[red, green]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let render_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxel Test Render Group"),
            layout: &pass.render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pass.meshlet_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pass.palette_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene_materials_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pass.material_map_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: pass.render_origin_buf.as_entire_binding(),
                },
            ],
        });
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel Test Color"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel Test Depth"),
            size: color.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let indirect_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel Test Indirect Read"),
            size: 2 * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let color_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxel Test Color Read"),
            size: 64 * 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut compute = encoder.begin_compute_pass(&Default::default());
            compute.set_pipeline(&pass.extract_pipeline);
            compute.set_bind_group(0, &pass.extract_bind_group, &[]);
            compute.dispatch_workgroups(2, 1, 1);
        }
        let stride = std::mem::size_of::<DrawIndexedIndirectArgs>() as u64;
        encoder.copy_buffer_to_buffer(
            &pass.staging_indirect_buf,
            0,
            &pass.indirect_buf,
            0,
            2 * stride,
        );
        encoder.copy_buffer_to_buffer(&pass.staging_indirect_buf, 0, &indirect_read, 0, 2 * stride);
        {
            let color_view = color.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let mut render = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Voxel Test Render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render.set_pipeline(&pass.render_pipeline);
            render.set_bind_group(0, &render_group, &[]);
            render.set_vertex_buffer(0, pass.vertex_buf.slice(..));
            render.set_vertex_buffer(1, pass.normal_buf.slice(..));
            render.set_index_buffer(pass.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            for slot in 0..2u64 {
                render.draw_indexed_indirect(&pass.indirect_buf, slot * stride);
            }
        }
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &color_read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            color.size(),
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        indirect_read.slice(..).map_async(wgpu::MapMode::Read, {
            let tx = tx.clone();
            move |result| tx.send(result).unwrap()
        });
        color_read
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();
        let indirect = indirect_read.slice(..).get_mapped_range().unwrap();
        let args: &[DrawIndexedIndirectArgs] = bytemuck::cast_slice(&indirect);
        assert_eq!(
            args[0].index_count, 1920,
            "shared block faces must be culled"
        );
        assert_eq!(
            args[1].index_count, 1920,
            "negative shared faces must be culled"
        );
        assert_eq!(args[1].first_instance, 1);
        drop(indirect);
        indirect_read.unmap();
        let pixels = color_read.slice(..).get_mapped_range().unwrap();
        let non_black = pixels
            .chunks_exact(4)
            .filter(|rgba| rgba[0] > 0 || rgba[1] > 0 || rgba[2] > 0)
            .count();
        assert!(
            non_black > 100,
            "voxel scene path must draw visible color; got {non_black} pixels"
        );
        let red_pixels = pixels
            .chunks_exact(4)
            .filter(|rgba| u16::from(rgba[0]) > u16::from(rgba[1]) * 2 && rgba[0] > 16)
            .count();
        let green_pixels = pixels
            .chunks_exact(4)
            .filter(|rgba| u16::from(rgba[1]) > u16::from(rgba[0]) * 2 && rgba[1] > 16)
            .count();
        assert!(
            red_pixels > 100 && green_pixels > 100,
            "both SceneDB material records must appear; red={red_pixels}, green={green_pixels}"
        );
        if let Ok(path) = std::env::var("HELIO_VOXEL_CAPTURE_RAW") {
            std::fs::write(path, &*pixels).expect("write requested raw RGBA capture");
        }
        drop(pixels);
        color_read.unmap();

        // Exercise the production smooth extractor and render pipeline on the
        // same canonical two-chunk data. Chunk 1 yields negative-X seam cells
        // to chunk 0, so the smooth boundary has one owner.
        for (slot, owner_mask) in [(0u32, 0xffu32), (1, 0x55)] {
            dirty[slot as usize].mode = VOXEL_MODE_SURFACE;
            dirty[slot as usize]._pad = owner_mask;
        }
        queue.write_buffer(&pass.dirty_brick_buf, 0, bytemuck::cast_slice(&dirty));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut compute = encoder.begin_compute_pass(&Default::default());
            compute.set_pipeline(&pass.extract_pipeline);
            compute.set_bind_group(0, &pass.extract_bind_group, &[]);
            compute.dispatch_workgroups(2, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &pass.staging_indirect_buf,
            0,
            &pass.indirect_buf,
            0,
            2 * stride,
        );
        encoder.copy_buffer_to_buffer(&pass.staging_indirect_buf, 0, &indirect_read, 0, 2 * stride);
        {
            let color_view = color.create_view(&Default::default());
            let depth_view = depth.create_view(&Default::default());
            let mut render = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Voxel Smooth Test Render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render.set_pipeline(&pass.render_pipeline);
            render.set_bind_group(0, &render_group, &[]);
            render.set_vertex_buffer(0, pass.vertex_buf.slice(..));
            render.set_vertex_buffer(1, pass.normal_buf.slice(..));
            render.set_index_buffer(pass.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            for slot in 0..2u64 {
                render.draw_indexed_indirect(&pass.indirect_buf, slot * stride);
            }
        }
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &color_read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            color.size(),
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        indirect_read.slice(..).map_async(wgpu::MapMode::Read, {
            let tx = tx.clone();
            move |result| tx.send(result).unwrap()
        });
        color_read
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();
        let indirect = indirect_read.slice(..).get_mapped_range().unwrap();
        let smooth_args: &[DrawIndexedIndirectArgs] = bytemuck::cast_slice(&indirect);
        assert!(
            smooth_args[0].index_count > 0 && smooth_args[1].index_count > 0,
            "smooth mode must extract both adjacent canonical chunks"
        );
        assert_eq!(smooth_args[1].first_instance, 1);
        drop(indirect);
        indirect_read.unmap();
        let pixels = color_read.slice(..).get_mapped_range().unwrap();
        let smooth_pixels = pixels
            .chunks_exact(4)
            .filter(|rgba| rgba[0] > 0 || rgba[1] > 0 || rgba[2] > 0)
            .count();
        assert!(
            smooth_pixels > 100,
            "smooth scene path must draw visible color; got {smooth_pixels} pixels"
        );
        if let Ok(path) = std::env::var("HELIO_VOXEL_CAPTURE_RAW") {
            std::fs::write(format!("{path}.smooth"), &*pixels).expect("write smooth RGBA capture");
        }
        drop(pixels);
        color_read.unmap();
        assert_eq!(
            pass.try_mark_dirty_with_mode(
                VOXEL_MESH_MAX_BRICKS,
                0,
                [0.0; 3],
                1.0,
                true,
                VOXEL_MODE_CUBES
            ),
            Err(VoxelDirtyError::SlotOutOfRange(VOXEL_MESH_MAX_BRICKS)),
        );
        pass.dirty_bricks
            .resize(VOXEL_MESH_MAX_DIRTY as usize, DirtyBrick::zeroed());
        assert_eq!(
            pass.try_mark_dirty_with_mode(0, 0, [0.0; 3], 1.0, true, VOXEL_MODE_CUBES),
            Err(VoxelDirtyError::DirtyListFull),
        );
        assert_eq!(pass.rejected_dirty_entries, 2);
    });
}
