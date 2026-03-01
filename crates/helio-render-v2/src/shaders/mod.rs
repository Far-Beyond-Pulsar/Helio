//! Shader module loading

mod include;
mod compiler;

pub use include::ShaderIncludeResolver;
pub use compiler::ShaderCompiler;

/// Shader library manages shader modules
pub struct ShaderLibrary {
    // TODO: Implement
}

impl ShaderLibrary {
    pub fn new() -> Self {
        Self {}
    }
}
