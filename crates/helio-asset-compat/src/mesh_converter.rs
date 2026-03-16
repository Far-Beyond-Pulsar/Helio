//! Vertex format conversion from SolidRS to Helio
//!
//! Converts SolidRS's rich vertex format (up to 8 UVs, 4 colors, optional tangents)
//! to Helio's compact PackedVertex format (32 bytes).

use helio_render_v2::mesh::PackedVertex;
use crate::Result;
use solid_rs::geometry::{Vertex, Topology};
use solid_rs::scene::Mesh;

/// Convert a SolidRS vertex to Helio's PackedVertex format
pub fn convert_vertex(v: &Vertex) -> PackedVertex {
    // Extract position (mandatory)
    let position = [v.position.x, v.position.y, v.position.z];

    // Extract normal (auto-generate if missing)
    let normal = if let Some(n) = v.normal {
        [n.x, n.y, n.z]
    } else {
        log::warn!("Vertex missing normal, using +Y");
        [0.0, 1.0, 0.0]
    };

    // Extract primary UV channel (default to 0,0 if missing)
    let tex_coords = if let Some(uv) = v.uvs[0] {
        [uv.x, uv.y]
    } else {
        [0.0, 0.0]
    };

    // Extract tangent (or generate from normal if missing)
    let tangent = if let Some(t) = v.tangent {
        [t.x, t.y, t.z]
    } else {
        normal_to_tangent(normal)
    };

    // Warn if we're dropping data
    if v.uvs.iter().skip(1).any(|uv| uv.is_some()) {
        log::warn!("Mesh has multiple UV channels - only UV0 is supported, others will be discarded");
    }
    if v.colors.iter().any(|c| c.is_some()) {
        log::warn!("Mesh has vertex colors - not yet supported, will be discarded");
    }

    PackedVertex::new_with_tangent(position, normal, tex_coords, tangent)
}

/// Convert a SolidRS mesh to Helio vertex/index buffers
pub fn convert_mesh(
    mesh: &Mesh,
) -> Result<(Vec<PackedVertex>, Vec<u32>)> {
    // Convert all vertices
    let vertices: Vec<PackedVertex> = mesh.vertices.iter()
        .map(convert_vertex)
        .collect();

    // Collect all indices from all primitives
    // TODO: In Phase 3, we'll separate by material to create multiple DrawCalls
    let mut indices = Vec::new();
    for primitive in &mesh.primitives {
        // Only support triangle lists for now
        if primitive.topology != Topology::TriangleList {
            log::warn!(
                "Primitive has topology {:?}, only TriangleList is supported - skipping",
                primitive.topology
            );
            continue;
        }

        indices.extend_from_slice(&primitive.indices);
    }

    Ok((vertices, indices))
}

/// Compute a tangent perpendicular to the normal (used when no tangent is provided)
fn normal_to_tangent(n: [f32; 3]) -> [f32; 3] {
    // Choose the axis least aligned with n to avoid degeneracy
    let up = if n[1].abs() < 0.9 { [0.0f32, 1.0, 0.0] } else { [1.0f32, 0.0, 0.0] };
    // cross(up, n) gives a vector perpendicular to n
    let t = [
        up[1]*n[2] - up[2]*n[1],
        up[2]*n[0] - up[0]*n[2],
        up[0]*n[1] - up[1]*n[0],
    ];
    let len = (t[0]*t[0] + t[1]*t[1] + t[2]*t[2]).sqrt().max(1e-8);
    [t[0]/len, t[1]/len, t[2]/len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_minimal_vertex() {
        use glam::{Vec2, Vec3};

        let v = Vertex {
            position: Vec3::new(1.0, 2.0, 3.0),
            normal: Some(Vec3::new(0.0, 1.0, 0.0)),
            tangent: None,
            colors: [None; 4],
            uvs: [None; 8],
            skin_weights: None,
        };

        let packed = convert_vertex(&v);
        assert_eq!(packed.position, [1.0, 2.0, 3.0]);
        assert_eq!(packed.tex_coords, [0.0, 0.0]);
    }
}
