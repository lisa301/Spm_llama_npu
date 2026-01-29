use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

fn default_rope_theta() -> f32 {
    10_000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3VlConfig {
    pub model_type: String,
    pub architectures: Option<Vec<String>>,

    /// Some checkpoints expose this at the top-level config.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    pub image_token_id: u32,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,

    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub max_position_embeddings: usize,

    /// Some checkpoints omit `lm_head.weight` and tie output projection to embeddings.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub num_position_embeddings: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,

    pub deepstack_visual_indexes: Option<Vec<usize>>,
}

impl Qwen3VlConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading configuration from {}", path.display());
        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    pub fn text(&self) -> crate::models::llama3::Config {
        crate::models::llama3::Config {
            hidden_size: self.text_config.hidden_size,
            intermediate_size: self.text_config.intermediate_size,
            vocab_size: self.text_config.vocab_size,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_attention_heads: self.text_config.num_attention_heads,
            num_key_value_heads: self.text_config.num_key_value_heads,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta,
            bos_token_id: self.text_config.bos_token_id,
            eos_token_id: self.text_config.eos_token_id,
            max_seq_len: self.text_config.max_position_embeddings,
        }
    }
}
