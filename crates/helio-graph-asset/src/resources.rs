//! Passes that exist only to own a resource: graph-declared textures, and
//! persistent textures and buffers that live across frames and rebuilds.
//!
//! The resource nodes of a graph asset compile to these. Persistent ones are
//! published by name through the reflected-binding contract, so a pass (a
//! shader node) binds one by declaring a variable of that name.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{PassContext, RenderPass, ResourceRegistry, Result};

/// A `'static` copy of `name`. Resource names are `&'static str` throughout
/// Helio; a name authored in a graph is leaked once per distinct value.
pub fn intern(name: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES.get_or_init(Default::default).lock().unwrap();
    if let Some(existing) = names.get(name) {
        return existing;
    }
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.insert(leaked);
    leaked
}

fn single(name: &'static str) -> &'static [&'static str] {
    static SINGLES: OnceLock<Mutex<HashMap<&'static str, &'static [&'static str]>>> =
        OnceLock::new();
    let mut singles = SINGLES.get_or_init(Default::default).lock().unwrap();
    *singles
        .entry(name)
        .or_insert_with(|| Box::leak(Box::new([name])))
}

/// A colour or depth texture the graph allocates and recycles, declared by
/// name so passes can write and read it.
pub struct TransientTexturePass {
    name: &'static str,
    writes: &'static [&'static str],
    format: ResourceFormat,
    size: ResourceSize,
    depth: bool,
}

impl TransientTexturePass {
    pub fn new(name: &str, format: ResourceFormat, size: ResourceSize, depth: bool) -> Self {
        let name = intern(name);
        Self {
            name,
            writes: single(name),
            format,
            size,
            depth,
        }
    }
}

impl RenderPass for TransientTexturePass {
    fn name(&self) -> &'static str {
        "GraphTexture"
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        if self.depth {
            builder.write_depth(self.name, self.size);
        } else {
            builder.write_color(self.name, self.format, self.size);
        }
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> Result<()> {
        Ok(())
    }
}

/// Texture shape of a [`PersistentTexturePass`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistentTextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

/// A texture allocated once and kept: history, accumulation, feedback. A graph
/// rebuilt at the same size (shader reload, a changed setting) hands it on
/// untouched; at a new size it starts empty.
pub struct PersistentTexturePass {
    name: &'static str,
    desc: PersistentTextureDesc,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl PersistentTexturePass {
    pub fn new(device: &wgpu::Device, name: &str, desc: PersistentTextureDesc) -> Self {
        let name = intern(name);
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST;
        let allowed = desc
            .format
            .guaranteed_format_features(device.features())
            .allowed_usages;
        if allowed.contains(wgpu::TextureUsages::STORAGE_BINDING) {
            usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(name),
            size: wgpu::Extent3d {
                width: desc.width.max(1),
                height: desc.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: desc.format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            name,
            desc,
            _texture: texture,
            view,
        }
    }

    pub fn desc(&self) -> PersistentTextureDesc {
        self.desc
    }
}

impl RenderPass for PersistentTexturePass {
    fn name(&self) -> &'static str {
        "GraphPersistentTexture"
    }

    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else {
            return false;
        };
        if previous.name != self.name || previous.desc != self.desc {
            return false;
        }
        std::mem::swap(&mut self._texture, &mut previous._texture);
        std::mem::swap(&mut self.view, &mut previous.view);
        true
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> Result<()> {
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut ResourceRegistry<'a>) {
        frame.write_texture_binding(self.name, &self.view, "GraphPersistentTexture");
    }
}

/// A buffer allocated once and kept: counters, particle state, accumulators.
/// Handed on across rebuilds when its size is unchanged.
pub struct PersistentBufferPass {
    name: &'static str,
    size: u64,
    buffer: wgpu::Buffer,
}

impl PersistentBufferPass {
    pub fn new(device: &wgpu::Device, name: &str, size: u64) -> Self {
        let name = intern(name);
        let size = size.max(4).next_multiple_of(4);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { name, size, buffer }
    }
}

impl RenderPass for PersistentBufferPass {
    fn name(&self) -> &'static str {
        "GraphPersistentBuffer"
    }

    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else {
            return false;
        };
        if previous.name != self.name || previous.size != self.size {
            return false;
        }
        std::mem::swap(&mut self.buffer, &mut previous.buffer);
        true
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> Result<()> {
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut ResourceRegistry<'a>) {
        frame.write_buffer_binding(self.name, &self.buffer, "GraphPersistentBuffer");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_returns_one_allocation_per_name() {
        let a = intern("history_color");
        let b = intern(&String::from("history_color"));
        assert!(std::ptr::eq(a, b));
        assert_ne!(intern("other"), a);
    }
}
