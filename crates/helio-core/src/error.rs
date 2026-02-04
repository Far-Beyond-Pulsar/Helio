use thiserror::Error;

#[derive(Error, Debug)]
pub enum HelioError {
    #[error("GPU device error: {0}")]
    GpuDeviceError(String),
    
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationError(String),
    
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, HelioError>;
