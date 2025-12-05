//! Error types for OmniAGI Core

use thiserror::Error;

/// Result type alias for OmniAGI operations
pub type Result<T> = std::result::Result<T, OmniError>;

/// Main error type for OmniAGI Core
#[derive(Error, Debug)]
pub enum OmniError {
    #[error("Model error: {0}")]
    Model(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
