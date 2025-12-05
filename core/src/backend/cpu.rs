//! CPU compute backend

use candle_core::Device;
use crate::error::Result;
use super::Backend;

/// CPU backend for inference
pub struct CpuBackend {
    device: Device,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "CPU"
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn is_available() -> bool {
        true // CPU is always available
    }
}
