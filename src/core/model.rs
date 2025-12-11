use log::*;

use crate::core::layer::KVCache;
use crate::profile;

use super::layer::{Linear, LlamaAttention, LlamaBlock, LlamaMLP, LlamaRMSNorm};
use super::shape::Shape;
use super::tensor::Tensor;

pub struct LlamaModel {
    pub embed_tokens: Tensor,
    pub layers: Vec<LlamaBlock>,
    pub norm: LlamaRMSNorm,
    pub lm_head: Linear,
}

impl LlamaModel {
    pub fn new(
        embed_tokens: Tensor,
        layers: Vec<LlamaBlock>,
        norm: LlamaRMSNorm,
        lm_head: Linear,
    ) -> Self {
        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }

    // 테스트용 더미 모델 생성자
    pub fn dummy(hidden_dim: usize, num_layers: usize) -> Self {
        let vocab_size = 10;
        let w_eye = Tensor::new(
            vec![1.0; hidden_dim * hidden_dim],
            Shape::new(vec![hidden_dim, hidden_dim]),
        ); // Shape mismatch fix needed for real identity
        let w_ones = Tensor::new(vec![1.0; hidden_dim], Shape::new(vec![hidden_dim]));

        // (간소화를 위해 모든 가중치를 1.0으로 채움)
        let create_linear = || {
            Linear::new(Tensor::new(
                vec![0.1; hidden_dim * hidden_dim],
                Shape::new(vec![hidden_dim, hidden_dim]),
            ))
        };

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(LlamaBlock {
                attn: LlamaAttention {
                    q_proj: create_linear(),
                    k_proj: create_linear(),
                    v_proj: create_linear(),
                    o_proj: create_linear(),
                    n_heads: 2,
                    cache: KVCache::new(2, hidden_dim / 2, 512),
                },
                mlp: LlamaMLP::new(create_linear(), create_linear(), create_linear()),
                input_norm: LlamaRMSNorm::new(w_ones.clone(), 1e-5),
                post_norm: LlamaRMSNorm::new(w_ones.clone(), 1e-5),
            });
        }

        let head_weight = Tensor::new(
            vec![0.1; hidden_dim * vocab_size],
            Shape::new(vec![hidden_dim, vocab_size]),
        );

        Self {
            embed_tokens: Tensor::new(
                vec![0.1; vocab_size * hidden_dim],
                Shape::new(vec![vocab_size, hidden_dim]),
            ),
            layers,
            norm: LlamaRMSNorm::new(w_ones, 1e-5),
            lm_head: Linear::new(head_weight),
        }
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Tensor {
        let mut h: Tensor;
        {
            profile!("4.1 LlamaModel cpy");
            h = x.clone();
            debug!("LlamaModel forward: input shape={:?}", h.shape().dims());
        }

        // 1. Layers
        {
            profile!("4.2 LlamaModel layers");
            for layer in &self.layers {
                debug!("layer: shape={:?}", h.shape().dims());
                h = layer.forward(&h, start_pos);
            }
        }

        // 2. Final Norm
        let h_norm: Tensor;
        {
            profile!("4.3 LlamaModel final norm");
            debug!("final norm: shape={:?}", h.shape().dims());
            h_norm = self.norm.forward(&h);
        }

        // 3. LM Head (Logits)
        {
            profile!("4.4 LlamaModel lm head");
            debug!("lm head: shape={:?}", h_norm.shape().dims());
            self.lm_head.forward(&h_norm)
        }
    }
}
