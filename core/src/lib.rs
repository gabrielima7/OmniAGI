//! # OmniAGI Core
//!
//! High-performance LLM inference engine for the OmniAGI cognitive operating system.
//!
//! This crate provides:
//! - Efficient model loading and inference
//! - Quantization support (GGUF, AWQ)
//! - Multi-backend support (CPU, CUDA, Metal)
//! - Python bindings via PyO3

pub mod engine;
pub mod backend;
pub mod error;

#[cfg(feature = "python")]
pub mod ffi;

pub use engine::{Model, InferenceConfig, GenerationOutput};
pub use error::{OmniError, Result};

/// Re-export candle types for convenience
pub mod prelude {
    pub use candle_core::{Device, Tensor, DType};
    pub use crate::engine::*;
    pub use crate::error::*;
}
