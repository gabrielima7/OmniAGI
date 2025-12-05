//! Inference configuration

use serde::{Deserialize, Serialize};

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    
    /// Temperature for sampling (0.0 = greedy, 1.0 = more random)
    pub temperature: f32,
    
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
    
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
    
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            seed: None,
            stop_sequences: Vec::new(),
        }
    }
}

impl InferenceConfig {
    /// Create a greedy decoding configuration
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            ..Default::default()
        }
    }
    
    /// Create a creative/high-temperature configuration
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 50,
            ..Default::default()
        }
    }
}
