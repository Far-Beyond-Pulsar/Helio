//! Source-level pipeline directives for shader-driven pass construction.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendDirective {
    Alpha,
    Additive,
    Replace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareDirective {
    Never,
    Less,
    LessEqual,
    Equal,
    GreaterEqual,
    Greater,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullDirective {
    None,
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyDirective {
    Point,
    Line,
    Triangle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PipelineDirectives {
    pub blend: Option<BlendDirective>,
    pub depth_compare: Option<CompareDirective>,
    pub depth_write: Option<bool>,
    pub cull: Option<CullDirective>,
    pub topology: Option<TopologyDirective>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectiveError {
    pub line: usize,
    pub message: String,
}

impl PipelineDirectives {
    pub fn primitive_state(&self) -> wgpu::PrimitiveState {
        wgpu::PrimitiveState {
            topology: match self.topology.unwrap_or(TopologyDirective::Triangle) {
                TopologyDirective::Point => wgpu::PrimitiveTopology::PointList,
                TopologyDirective::Line => wgpu::PrimitiveTopology::LineList,
                TopologyDirective::Triangle => wgpu::PrimitiveTopology::TriangleList,
            },
            cull_mode: match self.cull.unwrap_or(CullDirective::Back) {
                CullDirective::None => None,
                CullDirective::Front => Some(wgpu::Face::Front),
                CullDirective::Back => Some(wgpu::Face::Back),
            },
            ..Default::default()
        }
    }

    pub fn blend_state(&self) -> Option<wgpu::BlendState> {
        match self.blend {
            None | Some(BlendDirective::Replace) => None,
            Some(BlendDirective::Alpha) => Some(wgpu::BlendState::ALPHA_BLENDING),
            Some(BlendDirective::Additive) => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
        }
    }

    pub fn depth_stencil_state(
        &self,
        format: wgpu::TextureFormat,
    ) -> Option<wgpu::DepthStencilState> {
        let compare = self.depth_compare?;
        Some(wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(self.depth_write.unwrap_or(true)),
            depth_compare: Some(match compare {
                CompareDirective::Never => wgpu::CompareFunction::Never,
                CompareDirective::Less => wgpu::CompareFunction::Less,
                CompareDirective::LessEqual => wgpu::CompareFunction::LessEqual,
                CompareDirective::Equal => wgpu::CompareFunction::Equal,
                CompareDirective::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
                CompareDirective::Greater => wgpu::CompareFunction::Greater,
                CompareDirective::Always => wgpu::CompareFunction::Always,
            }),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        })
    }
}

impl std::fmt::Display for DirectiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shader directive line {}: {}", self.line, self.message)
    }
}
impl std::error::Error for DirectiveError {}

/// Parses `//!blend`, `//!depth`, `//!cull`, and `//!topology` comments.
pub fn parse(source: &str) -> Result<PipelineDirectives, DirectiveError> {
    let mut out = PipelineDirectives::default();
    for (index, line) in source.lines().enumerate() {
        let line_number = index + 1;
        let Some(rest) = line.trim().strip_prefix("//!") else {
            continue;
        };
        let mut words = rest.split_whitespace();
        let Some(kind) = words.next() else { continue };
        let values: Vec<_> = words.collect();
        let value = |name: &str| {
            values.first().copied().ok_or_else(|| DirectiveError {
                line: line_number,
                message: format!("{name} requires a value"),
            })
        };
        let invalid = |message: String| DirectiveError {
            line: line_number,
            message,
        };
        match kind {
            "blend" => {
                out.blend = Some(match value("blend")? {
                    "alpha" => BlendDirective::Alpha,
                    "additive" => BlendDirective::Additive,
                    "replace" => BlendDirective::Replace,
                    other => return Err(invalid(format!("unknown blend mode '{other}'"))),
                })
            }
            "depth" => {
                out.depth_compare = Some(match value("depth")? {
                    "never" => CompareDirective::Never,
                    "less" => CompareDirective::Less,
                    "less_equal" => CompareDirective::LessEqual,
                    "equal" => CompareDirective::Equal,
                    "greater_equal" => CompareDirective::GreaterEqual,
                    "greater" => CompareDirective::Greater,
                    "always" => CompareDirective::Always,
                    other => return Err(invalid(format!("unknown depth compare '{other}'"))),
                });
                if let Some(mode) = values.get(1) {
                    out.depth_write = Some(match *mode {
                        "write" => true,
                        "no_write" => false,
                        other => {
                            return Err(invalid(format!("unknown depth write mode '{other}'")))
                        }
                    });
                }
            }
            "cull" => {
                out.cull = Some(match value("cull")? {
                    "none" => CullDirective::None,
                    "front" => CullDirective::Front,
                    "back" => CullDirective::Back,
                    other => return Err(invalid(format!("unknown cull mode '{other}'"))),
                })
            }
            "topology" => {
                out.topology = Some(match value("topology")? {
                    "point" => TopologyDirective::Point,
                    "line" => TopologyDirective::Line,
                    "triangle" => TopologyDirective::Triangle,
                    other => return Err(invalid(format!("unknown topology '{other}'"))),
                })
            }
            _ => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_fixed_function_directives() {
        let d = parse(
            "//!blend alpha\n//!depth less_equal no_write\n//!cull back\n//!topology triangle",
        )
        .unwrap();
        assert_eq!(d.blend, Some(BlendDirective::Alpha));
        assert_eq!(d.depth_compare, Some(CompareDirective::LessEqual));
        assert_eq!(d.depth_write, Some(false));
        assert_eq!(d.cull, Some(CullDirective::Back));
        assert_eq!(d.topology, Some(TopologyDirective::Triangle));
    }
    #[test]
    fn reports_malformed_directives() {
        assert_eq!(parse("//!depth invalid").unwrap_err().line, 1);
    }

    #[test]
    fn directives_compile_to_wgpu_fixed_function_state() {
        let directives =
            parse("//!blend additive\n//!depth greater no_write\n//!cull front\n//!topology line")
                .unwrap();
        assert_eq!(
            directives.primitive_state().topology,
            wgpu::PrimitiveTopology::LineList
        );
        assert_eq!(
            directives.primitive_state().cull_mode,
            Some(wgpu::Face::Front)
        );
        assert!(directives.blend_state().is_some());
        let depth = directives
            .depth_stencil_state(wgpu::TextureFormat::Depth32Float)
            .unwrap();
        assert_eq!(depth.depth_write_enabled, Some(false));
        assert_eq!(depth.depth_compare, Some(wgpu::CompareFunction::Greater));
    }
}
