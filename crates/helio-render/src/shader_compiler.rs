use std::path::Path;

pub struct ShaderCompiler {
    
}

impl ShaderCompiler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compile_from_file(&self, path: &Path) -> Result<String, String> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read shader: {}", e))?;
        self.compile_from_source(&source)
    }

    pub fn compile_from_source(&self, source: &str) -> Result<String, String> {
        Ok(source.to_string())
    }
}

impl Default for ShaderCompiler {
    fn default() -> Self {
        Self::new()
    }
}
