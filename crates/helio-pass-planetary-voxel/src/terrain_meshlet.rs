use crate::{GpuTerrainMeshlet, GpuTerrainVertex};
use bytemuck::{Pod, Zeroable};

/// Fixed GPU-build partition. Keeping this below the public 96-triangle
/// contract guarantees the 64-vertex limit even when no triangle shares a
/// vertex, without rewriting or welding Transvoxel output.
pub const TERRAIN_MESHLET_BUILD_TRIANGLES: u32 = 21;
pub const TERRAIN_MESHLET_BUILD_INDICES: u32 = TERRAIN_MESHLET_BUILD_TRIANGLES * 3;

/// Conservative culling data for one terrain meshlet.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuTerrainMeshletBounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_apex: [f32; 3],
    pub cone_cutoff: f32,
    pub cone_axis: [f32; 3],
    pub _pad: f32,
}

/// Maps a compact indirect draw back to its source page and meshlet.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTerrainDraw {
    pub page_slot: u32,
    pub meshlet_index: u32,
    pub surface_kind: u32,
    pub lod: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTerrainCullUniforms {
    pub max_meshlets_per_bank: u32,
    pub draw_capacity: u32,
    pub surface_kind: u32,
    pub _pad: u32,
}

/// Shared counters written by the regular and transition meshlet cull passes.
/// The first two words can be consumed directly as indirect draw counts.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuTerrainCullCounters {
    pub regular_draws: u32,
    pub transition_draws: u32,
    pub overflow: u32,
    pub stale: u32,
    pub frustum_rejects: u32,
    pub cone_rejects: u32,
    pub invalid_candidates: u32,
    pub _reserved: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainMeshletBuild {
    pub descriptor: GpuTerrainMeshlet,
    pub bounds: GpuTerrainMeshletBounds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TerrainMeshletBuildError {
    #[error("terrain meshlet indices must contain complete triangles")]
    IncompleteTriangle,
    #[error("terrain meshlet index {index} is outside {vertex_count} vertices")]
    IndexOutOfBounds { index: u32, vertex_count: u32 },
    #[error("terrain meshlet vertex {index} contains a non-finite position")]
    NonFinitePosition { index: u32 },
    #[error("terrain meshlet range exceeds the u32 GPU address space")]
    AddressOverflow,
}

pub const fn max_meshlets_for_indices(max_indices: u32) -> u32 {
    max_indices.div_ceil(TERRAIN_MESHLET_BUILD_INDICES)
}

/// Builds the same deterministic fixed partition used by the GPU publisher.
///
/// The descriptors reference the original vertex and index streams. No
/// welding, remapping, material coalescing, or position-only identity is
/// introduced by meshlet publication.
pub fn build_terrain_meshlets(
    vertices: &[GpuTerrainVertex],
    indices: &[u32],
    first_index: u32,
    first_vertex: u32,
    first_bounds: u32,
    generation: u64,
    flags: u32,
) -> Result<Vec<TerrainMeshletBuild>, TerrainMeshletBuildError> {
    if !indices.len().is_multiple_of(3) {
        return Err(TerrainMeshletBuildError::IncompleteTriangle);
    }

    let vertex_count =
        u32::try_from(vertices.len()).map_err(|_| TerrainMeshletBuildError::AddressOverflow)?;
    for (index, vertex) in vertices.iter().enumerate() {
        if !vertex
            .position
            .iter()
            .all(|component| component.is_finite())
        {
            return Err(TerrainMeshletBuildError::NonFinitePosition {
                index: u32::try_from(index)
                    .map_err(|_| TerrainMeshletBuildError::AddressOverflow)?,
            });
        }
    }
    for &index in indices {
        if index >= vertex_count {
            return Err(TerrainMeshletBuildError::IndexOutOfBounds {
                index,
                vertex_count,
            });
        }
    }

    let index_count =
        u32::try_from(indices.len()).map_err(|_| TerrainMeshletBuildError::AddressOverflow)?;
    let mut builds = Vec::with_capacity(max_meshlets_for_indices(index_count) as usize);
    for (meshlet_index, chunk) in indices
        .chunks(TERRAIN_MESHLET_BUILD_INDICES as usize)
        .enumerate()
    {
        let meshlet_index =
            u32::try_from(meshlet_index).map_err(|_| TerrainMeshletBuildError::AddressOverflow)?;
        let chunk_offset = meshlet_index
            .checked_mul(TERRAIN_MESHLET_BUILD_INDICES)
            .ok_or(TerrainMeshletBuildError::AddressOverflow)?;
        let index_count =
            u32::try_from(chunk.len()).map_err(|_| TerrainMeshletBuildError::AddressOverflow)?;

        builds.push(TerrainMeshletBuild {
            descriptor: GpuTerrainMeshlet {
                first_index: first_index
                    .checked_add(chunk_offset)
                    .ok_or(TerrainMeshletBuildError::AddressOverflow)?,
                index_count,
                first_vertex,
                vertex_count: unique_index_count(chunk),
                bounds_offset: first_bounds
                    .checked_add(meshlet_index)
                    .ok_or(TerrainMeshletBuildError::AddressOverflow)?,
                generation_low: generation as u32,
                generation_high: (generation >> 32) as u32,
                _pad: flags,
            },
            bounds: compute_bounds(vertices, chunk),
        });
    }
    Ok(builds)
}

/// Exact perspective cone predicate shared by CPU validation and GPU culling.
/// A disabled or degenerate cone has a cutoff of one and cannot reject.
pub fn perspective_cone_reject(
    bounds: &GpuTerrainMeshletBounds,
    camera_position: [f32; 3],
) -> bool {
    let view = sub(bounds.cone_apex, camera_position);
    let Some(view) = normalize(view) else {
        return false;
    };
    dot(view, bounds.cone_axis) >= bounds.cone_cutoff
}

fn unique_index_count(indices: &[u32]) -> u32 {
    let mut unique = [u32::MAX; 64];
    let mut count = 0usize;
    for &index in indices {
        if !unique[..count].contains(&index) {
            unique[count] = index;
            count += 1;
        }
    }
    count as u32
}

fn compute_bounds(vertices: &[GpuTerrainVertex], indices: &[u32]) -> GpuTerrainMeshletBounds {
    let first = vertices[indices[0] as usize].position;
    let mut min = first;
    let mut max = first;
    for &index in indices {
        let position = vertices[index as usize].position;
        min = component_min(min, position);
        max = component_max(max, position);
    }
    let center = scale(add(min, max), 0.5);
    let radius = indices
        .iter()
        .map(|&index| length(sub(vertices[index as usize].position, center)))
        .fold(0.0_f32, f32::max);

    let normals: Vec<[f32; 3]> = indices
        .chunks_exact(3)
        .filter_map(|triangle| {
            let a = vertices[triangle[0] as usize].position;
            let b = vertices[triangle[1] as usize].position;
            let c = vertices[triangle[2] as usize].position;
            normalize(cross(sub(b, a), sub(c, a)))
        })
        .collect();
    let Some(axis) = normalize(
        normals
            .iter()
            .copied()
            .fold([0.0; 3], add),
    ) else {
        return disabled_bounds(center, radius);
    };
    let min_dot = normals
        .iter()
        .map(|&normal| dot(normal, axis))
        .fold(1.0_f32, f32::min);
    if min_dot <= 0.1 {
        return disabled_bounds(center, radius);
    }

    let mut apex_distance = 0.0_f32;
    for triangle in indices.chunks_exact(3) {
        let a = vertices[triangle[0] as usize].position;
        let b = vertices[triangle[1] as usize].position;
        let c = vertices[triangle[2] as usize].position;
        let Some(normal) = normalize(cross(sub(b, a), sub(c, a))) else {
            continue;
        };
        let denominator = dot(axis, normal);
        if denominator > 0.0 {
            apex_distance = apex_distance.max(dot(sub(center, a), normal) / denominator);
        }
    }

    GpuTerrainMeshletBounds {
        center,
        radius,
        cone_apex: sub(center, scale(axis, apex_distance)),
        // Raising the cutoff is conservative and absorbs CPU/GPU rounding.
        cone_cutoff: ((1.0 - min_dot * min_dot).max(0.0).sqrt() + 1.0e-4).min(1.0),
        cone_axis: axis,
        _pad: 0.0,
    }
}

fn disabled_bounds(center: [f32; 3], radius: f32) -> GpuTerrainMeshletBounds {
    GpuTerrainMeshletBounds {
        center,
        radius,
        cone_apex: center,
        cone_cutoff: 1.0,
        cone_axis: [0.0; 3],
        _pad: 0.0,
    }
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(value: [f32; 3], factor: f32) -> [f32; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(value: [f32; 3]) -> f32 {
    dot(value, value).sqrt()
}

fn normalize(value: [f32; 3]) -> Option<[f32; 3]> {
    let magnitude = length(value);
    (magnitude > f32::EPSILON).then(|| scale(value, magnitude.recip()))
}

fn component_min(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

fn component_max(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(position: [f32; 3], material: u32, flags: u32) -> GpuTerrainVertex {
        GpuTerrainVertex {
            position,
            material,
            normal: [1.0, 0.0, 0.0],
            flags,
        }
    }

    #[test]
    fn fixed_partition_preserves_every_triangle_and_generation() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for triangle in 0..50_u32 {
            let base = vertices.len() as u32;
            vertices.extend([
                vertex([triangle as f32, 0.0, 0.0], triangle, 1),
                vertex([triangle as f32, 1.0, 0.0], triangle, 2),
                vertex([triangle as f32, 0.0, 1.0], triangle, 3),
            ]);
            indices.extend([base, base + 1, base + 2]);
        }

        let generation = 0x0123_4567_89ab_cdef;
        let builds =
            build_terrain_meshlets(&vertices, &indices, 100, 200, 300, generation, 7).unwrap();
        assert_eq!(builds.len(), 3);
        assert_eq!(
            builds
                .iter()
                .map(|build| build.descriptor.index_count)
                .sum::<u32>(),
            indices.len() as u32
        );
        for (meshlet, build) in builds.iter().enumerate() {
            assert_eq!(
                build.descriptor.first_index,
                100 + meshlet as u32 * TERRAIN_MESHLET_BUILD_INDICES
            );
            assert_eq!(build.descriptor.first_vertex, 200);
            assert!(build.descriptor.vertex_count <= 64);
            assert!(build.descriptor.index_count / 3 <= 96);
            assert_eq!(build.descriptor.bounds_offset, 300 + meshlet as u32);
            assert_eq!(build.descriptor.generation_low, generation as u32);
            assert_eq!(build.descriptor.generation_high, (generation >> 32) as u32);
            assert_eq!(build.descriptor._pad, 7);
        }
    }

    #[test]
    fn publication_does_not_weld_position_equal_seam_vertices() {
        let vertices = [
            vertex([0.0, 0.0, 0.0], 1, 0),
            vertex([0.0, 0.0, 0.0], 2, 1),
            vertex([0.0, 1.0, 0.0], 1, 0),
        ];
        let build = build_terrain_meshlets(&vertices, &[0, 1, 2], 0, 0, 0, 1, 0).unwrap()[0];
        assert_eq!(build.descriptor.vertex_count, 3);
    }

    #[test]
    fn sphere_contains_all_referenced_vertices() {
        let vertices = [
            vertex([-4.0, 2.0, 0.5], 1, 0),
            vertex([8.0, -3.0, 1.5], 1, 0),
            vertex([1.0, 7.0, -6.0], 1, 0),
        ];
        let build = build_terrain_meshlets(&vertices, &[0, 1, 2], 0, 0, 0, 1, 0).unwrap()[0];
        for vertex in vertices {
            assert!(
                length(sub(vertex.position, build.bounds.center)) <= build.bounds.radius + 1.0e-5
            );
        }
    }

    #[test]
    fn planar_cone_rejects_only_the_back_side() {
        let vertices = [
            vertex([0.0, 0.0, 0.0], 1, 0),
            vertex([0.0, 1.0, 0.0], 1, 0),
            vertex([0.0, 0.0, 1.0], 1, 0),
        ];
        let build = build_terrain_meshlets(&vertices, &[0, 1, 2], 0, 0, 0, 1, 0).unwrap()[0];
        assert!(perspective_cone_reject(&build.bounds, [-10.0, 0.25, 0.25]));
        assert!(!perspective_cone_reject(&build.bounds, [10.0, 0.25, 0.25]));
    }
}
