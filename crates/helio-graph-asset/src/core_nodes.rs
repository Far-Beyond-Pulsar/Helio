//! The nodes every graph has: the output, host-supplied inputs, and the
//! resource nodes (transient and persistent textures, persistent buffers).

use helio_core::graph::{ResourceFormat, ResourceSize};
use wgpu::TextureFormat;

use crate::asset::Params;
use crate::pins::{PinDesc, PinType};
use crate::registry::{NodeBuild, NodeDescriptor, NodePins, NodeRegistry, ParamDesc, ParamType};
use crate::resources::{
    intern, PersistentBufferPass, PersistentTextureDesc, PersistentTexturePass,
    TransientTexturePass,
};

pub const ON_FRAME_START: &str = "core.on_frame_start";
pub const OUTPUT: &str = "core.output";
pub const BUFFER_GET: &str = "core.buffer_get";
pub const BUFFER_SET: &str = "core.buffer_set";
pub const EXTERNAL: &str = "core.external";
pub const TEXTURE: &str = "core.texture";
pub const DEPTH: &str = "core.depth";
pub const PERSISTENT_TEXTURE: &str = "core.persistent_texture";
pub const PERSISTENT_BUFFER: &str = "core.persistent_buffer";

const FORMATS: &[(&str, TextureFormat)] = &[
    ("rgba16float", TextureFormat::Rgba16Float),
    ("rgba8unorm", TextureFormat::Rgba8Unorm),
    ("rgba8unorm_srgb", TextureFormat::Rgba8UnormSrgb),
    ("rg16float", TextureFormat::Rg16Float),
    ("r16float", TextureFormat::R16Float),
    ("r32float", TextureFormat::R32Float),
    ("r32uint", TextureFormat::R32Uint),
    ("r8unorm", TextureFormat::R8Unorm),
];
const SIZES: &[&str] = &["internal", "half", "quarter", "output"];

fn format_names() -> Vec<&'static str> {
    FORMATS.iter().map(|(name, _)| *name).collect()
}

fn parse_format(name: &str) -> Result<TextureFormat, String> {
    FORMATS
        .iter()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, format)| *format)
        .ok_or_else(|| format!("unknown texture format `{name}`"))
}

fn transient_size(name: &str) -> Result<ResourceSize, String> {
    Ok(match name {
        "internal" => ResourceSize::MatchSurface,
        "half" => ResourceSize::ScaledInternal { divisor: 2 },
        "quarter" => ResourceSize::ScaledInternal { divisor: 4 },
        "output" => ResourceSize::Output,
        other => return Err(format!("unknown size `{other}`")),
    })
}

fn resource_name(build: &NodeBuild<'_, '_>) -> Result<String, String> {
    let name = build.text("name");
    if name.trim().is_empty() {
        return Err("a resource needs a name".into());
    }
    Ok(name)
}

/// One output pin named by the node's `name` parameter.
fn named_output(ty: PinType) -> impl Fn(&Params) -> NodePins + Send + Sync {
    move |params| {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        NodePins {
            inputs: Vec::new(),
            outputs: vec![PinDesc::resource(ty, name)],
        }
    }
}

/// The pin of a buffer node: the buffer its `buffer` parameter names. An empty
/// resource (no valid buffer yet) is reported when the graph is flattened.
fn buffer_pin(params: &Params) -> PinDesc {
    let (ty, name) = params
        .get("buffer")
        .and_then(|value| value.as_str())
        .and_then(crate::buffers::parse_entry)
        .unwrap_or((PinType::TEXTURE, String::new()));
    PinDesc::resource(ty, name)
}

pub fn register(registry: &mut NodeRegistry) {
    registry.register(
        NodeDescriptor::new(
            BUFFER_GET,
            "Get Buffer",
            "Buffers",
            "Offers a buffer wherever it is placed. Place as many as you like; they read the latest Set of the buffer before the node using them.",
            |_| Ok(()),
        )
        .with_param(ParamDesc::text("buffer", "Buffer", "tex:pre_aa"))
        .with_dynamic_pins(|params| NodePins { inputs: Vec::new(), outputs: vec![buffer_pin(params)] })
        .with_data_only(),
    );
    registry.register(
        NodeDescriptor::new(
            BUFFER_SET,
            "Set Buffer",
            "Buffers",
            "Names the value wired into it as a buffer, so Get Buffer nodes can read it elsewhere. Compiles away.",
            |_| Ok(()),
        )
        .with_param(ParamDesc::text("buffer", "Buffer", "tex:pre_aa"))
        .with_dynamic_pins(|params| NodePins { inputs: vec![buffer_pin(params)], outputs: Vec::new() })
        .with_data_only(),
    );

    registry.register(
        NodeDescriptor::new(
            ON_FRAME_START,
            "On Frame Start",
            "Graph",
            "Where a frame begins. Every pass that executes is chained from here, in the order the flow wires give.",
            |_| Ok(()),
        )
        .with_pins(Vec::new(), vec![PinDesc::flow("out")]),
    );

    registry.register(
        NodeDescriptor::new(
            OUTPUT,
            "Output",
            "Graph",
            "The end of the frame. The last pass in the chain flows into it.",
            |_| Ok(()),
        )
        .with_pins(vec![PinDesc::flow("in")], Vec::new()),
    );

    registry.register(
        NodeDescriptor::new(
            EXTERNAL,
            "Scene Input",
            "Graph",
            "A resource the renderer supplies every frame (material textures, the environment).",
            |build| {
                let name = resource_name(build)?;
                build.graph.declare_external_input(intern(&name));
                Ok(())
            },
        )
        .with_param(ParamDesc::text("name", "Resource", "render_environment"))
        .with_dynamic_pins(named_output(PinType::SCENE)),
    );

    registry.register(
        NodeDescriptor::new(
            TEXTURE,
            "Texture",
            "Resources",
            "A colour texture the graph allocates and recycles between passes.",
            |build| {
                let name = resource_name(build)?;
                let format = ResourceFormat::from(parse_format(&build.text("format"))?);
                let size = transient_size(&build.text("size"))?;
                build.graph.add_pass(Box::new(TransientTexturePass::new(
                    &name, format, size, false,
                )));
                Ok(())
            },
        )
        .with_param(ParamDesc::text("name", "Name", "my_texture"))
        .with_param(ParamDesc::choice(
            "format",
            "Format",
            &format_names(),
            "rgba16float",
        ))
        .with_param(ParamDesc::choice("size", "Size", SIZES, "internal"))
        .with_dynamic_pins(named_output(PinType::TEXTURE)),
    );

    registry.register(
        NodeDescriptor::new(
            DEPTH,
            "Depth Texture",
            "Resources",
            "A depth texture the graph allocates and recycles between passes.",
            |build| {
                let name = resource_name(build)?;
                let size = transient_size(&build.text("size"))?;
                build.graph.add_pass(Box::new(TransientTexturePass::new(
                    &name,
                    ResourceFormat::Depth32Float,
                    size,
                    true,
                )));
                Ok(())
            },
        )
        .with_param(ParamDesc::text("name", "Name", "my_depth"))
        .with_param(ParamDesc::choice("size", "Size", SIZES, "internal"))
        .with_dynamic_pins(named_output(PinType::DEPTH)),
    );

    registry.register(
        NodeDescriptor::new(
            PERSISTENT_TEXTURE,
            "Persistent Texture",
            "Resources",
            "A texture kept across frames (history, accumulation, feedback). Survives graph rebuilds at the same size.",
            |build| {
                let name = resource_name(build)?;
                let format = parse_format(&build.text("format"))?;
                let (iw, ih) = build.env.internal_size();
                let (width, height) = match build.text("size").as_str() {
                    "internal" => (iw, ih),
                    "half" => (iw / 2, ih / 2),
                    "quarter" => (iw / 4, ih / 4),
                    "output" => (build.env.config.width, build.env.config.height),
                    other => return Err(format!("unknown size `{other}`")),
                };
                let desc = PersistentTextureDesc { width: width.max(1), height: height.max(1), format };
                build.graph.add_pass(Box::new(PersistentTexturePass::new(build.env.device, &name, desc)));
                Ok(())
            },
        )
        .with_param(ParamDesc::text("name", "Name", "history_color"))
        .with_param(ParamDesc::choice("format", "Format", &format_names(), "rgba16float"))
        .with_param(ParamDesc::choice("size", "Size", SIZES, "internal"))
        .with_dynamic_pins(named_output(PinType::PERSISTENT_TEXTURE)),
    );

    registry.register(
        NodeDescriptor::new(
            PERSISTENT_BUFFER,
            "Persistent Buffer",
            "Resources",
            "A GPU buffer kept across frames (counters, particle state, accumulators). Survives graph rebuilds at the same size.",
            |build| {
                let name = resource_name(build)?;
                let size = u64::try_from(build.int("size_bytes"))
                    .map_err(|_| "buffer size must not be negative".to_string())?;
                build.graph.add_pass(Box::new(PersistentBufferPass::new(build.env.device, &name, size)));
                Ok(())
            },
        )
        .with_param(ParamDesc::text("name", "Name", "my_buffer"))
        .with_param(ParamDesc::new(
            "size_bytes",
            "Size (bytes)",
            ParamType::Int { min: 4, max: 1 << 30 },
            serde_json::Value::from(65536),
        ))
        .with_dynamic_pins(named_output(PinType::PERSISTENT_BUFFER)),
    );
}
