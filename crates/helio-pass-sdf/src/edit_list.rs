//! SDF edit list — ordered list of shape + transform + boolean operation.

use super::primitives::{SdfShapeType, SdfShapeParams};
use glam::Mat4;
use bytemuck::{Pod, Zeroable};

/// Boolean operation applied between an edit and the accumulated SDF.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    Union        = 0,
    Subtraction  = 1,
    Intersection = 2,
}

/// CPU-side SDF edit (user-friendly).
#[derive(Clone, Debug)]
pub struct SdfEdit {
    pub shape:        SdfShapeType,
    pub op:           BooleanOp,
    /// World-to-local transform (set to Mat4::IDENTITY for world-space edits).
    pub transform:    Mat4,
    pub params:       SdfShapeParams,
    pub blend_radius: f32,
}

/// GPU-side SDF edit (96 bytes, 16-byte aligned).
/// Layout must match the WGSL `SdfEdit` struct exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuSdfEdit {
    /// World-to-local transform (inverse of the object transform).
    pub transform:    [f32; 16],
    pub shape_type:   u32,
    pub boolean_op:   u32,
    pub blend_radius: f32,
    pub _pad0:        u32,
    pub params:       SdfShapeParams,
}

impl SdfEdit {
    pub fn to_gpu(&self) -> GpuSdfEdit {
        let inv = self.transform.inverse();
        GpuSdfEdit {
            transform:    inv.to_cols_array(),
            shape_type:   self.shape as u32,
            boolean_op:   self.op as u32,
            blend_radius: self.blend_radius,
            _pad0:        0,
            params:       self.params,
        }
    }
}

/// Ordered list of SDF edits with dirty tracking for GPU upload.
pub struct SdfEditList {
    edits:      Vec<SdfEdit>,
    dirty:      bool,
    generation: u64,
}

impl SdfEditList {
    pub fn new() -> Self {
        Self { edits: Vec::new(), dirty: true, generation: 0 }
    }

    /// Appends `edit` and returns its stable index.
    pub fn add(&mut self, edit: SdfEdit) -> usize {
        let idx = self.edits.len();
        self.edits.push(edit);
        self.dirty = true;
        self.generation += 1;
        idx
    }

    pub fn remove(&mut self, index: usize) {
        self.edits.remove(index);
        self.dirty = true;
        self.generation += 1;
    }

    pub fn set(&mut self, index: usize, edit: SdfEdit) {
        self.edits[index] = edit;
        self.dirty = true;
        self.generation += 1;
    }

    pub fn clear(&mut self) {
        self.edits.clear();
        self.dirty = true;
        self.generation += 1;
    }

    pub fn len(&self)        -> usize { self.edits.len() }
    pub fn is_empty(&self)   -> bool  { self.edits.is_empty() }
    pub fn is_dirty(&self)   -> bool  { self.dirty }
    pub fn generation(&self) -> u64   { self.generation }
    pub fn edits(&self)      -> &[SdfEdit] { &self.edits }

    /// Build the GPU buffer contents and clear the dirty flag.
    pub fn flush_gpu_data(&mut self) -> Vec<GpuSdfEdit> {
        self.dirty = false;
        self.edits.iter().map(|e| e.to_gpu()).collect()
    }
}

impl Default for SdfEditList {
    fn default() -> Self { Self::new() }
}
