use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    #[serde(rename = "num_hidden_layers")]
    pub num_layers: usize,
    #[serde(rename = "num_attention_heads")]
    pub num_heads: usize,
    #[serde(rename = "num_key_value_heads")]
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_rope_theta() -> f32 {
    500000.0
}
