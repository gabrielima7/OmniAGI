//! Compute backends for inference
//!
//! Supports CPU and GPU (CUDA/Metal) backends.

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "metal")]
mod metal;

pub use cpu::CpuBackend;

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

#[cfg(feature = "metal")]
pub use metal::MetalBackend;

use candle_core::Device;
use crate::error::Result;

/// Trait for compute backends
pub trait Backend: Send + Sync {
    /// Get the name of this backend
    fn name(&self) -> &str;
    
    /// Get the candle device for this backend
    fn device(&self) -> &Device;
    
    /// Check if this backend is available on the current system
    fn is_available() -> bool where Self: Sized;
}

/// Automatically select the best available backend
pub fn auto_select_backend() -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    if CudaBackend::is_available() {
        return Ok(Box::new(CudaBackend::new(0)?));
    }
    
    #[cfg(feature = "metal")]
    if MetalBackend::is_available() {
        return Ok(Box::new(MetalBackend::new()?));
    }
    
    Ok(Box::new(CpuBackend::new()))
}
