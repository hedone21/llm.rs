use log::*;

use crate::backend::Device;
use crate::profile;

use super::layer::{Linear, LlamaBlock, LlamaRMSNorm};
use super::tensor::Tensor;

pub struct LlamaModel {
    pub embed_tokens: Tensor,
    pub layers: Vec<LlamaBlock>,
    pub norm: LlamaRMSNorm,
    pub lm_head: Linear,
}

impl LlamaModel {
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
            let h_norm = self.norm.forward(&h);
            self.lm_head.forward(&h_norm)
        }
    }

    pub fn to_device(self, device: Device) -> Self {
        info!("Moving LlamaModel to {:?}", device);

        let embed_tokens = self.embed_tokens.to_device(device);
        let norm = self.norm.to_device(device);
        let lm_head = self.lm_head.to_device(device);

        // 레이어들은 병렬로 처리하지 않고 순차적으로 이동 (로딩 타임이므로)
        let layers = self
            .layers
            .into_iter()
            .map(|layer| layer.to_device(device))
            .collect();

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }
}
