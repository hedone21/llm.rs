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
