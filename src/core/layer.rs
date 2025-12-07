use std::cell::RefCell;

use crate::core::shape::Shape;

use super::tensor::Tensor;

pub struct Linear {
    weight_transposed: Vec<f32>,
    pub weight: Tensor,
}

impl Linear {
    // 가중치 텐서를 받아서 레이어 생성
    fn new(original_data: &[f32], out_features: usize, in_features: usize) -> Self {
        // [중요] 여기서 딱 한 번만 무거운 재배열 작업을 수행!
        let weight_transposed =
            transpose_and_make_contiguous(original_data, out_features, in_features);

        Self {
            weight_transposed,
            shape: (in_features, out_features), // Shape도 뒤집어서 저장
        }
    }

    // 순전파 (Forward Pass): 입력 X와 가중치 W를 곱함
    pub fn forward(&self, x: &Tensor) -> Tensor {
        matmul(input, &self.weight_transposed)
    }
}

pub struct LlamaMLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl LlamaMLP {
    pub fn new(gate: Linear, up: Linear, down: Linear) -> Self {
        Self {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Gate Projection: 정보의 흐름을 제어할 신호 생성
        // Shape: [Batch, Hidden] -> [Batch, Intermediate]
        let gate = self.gate_proj.forward(x);

        // 2. Up Projection: 실제 정보를 확장
        // Shape: [Batch, Hidden] -> [Batch, Intermediate]
        let up = self.up_proj.forward(x);

        // 3. SwiGLU Operation: (SiLU(Gate) * Up)
        // Gate에 비선형성(Silu)을 가하고, Up 정보와 요소별 곱셈(Mul)
        let gate_act = gate.silu();
        let fused = gate_act.mul(&up);

        // 4. Down Projection: 확장된 정보를 다시 원래 차원으로 압축
        // Shape: [Batch, Intermediate] -> [Batch, Hidden]
        self.down_proj.forward(&fused)
    }
}

// KV Cache (단순 Vec<f32> 기반)
pub struct KVCache {
    pub k: RefCell<Tensor>, // Interior Mutability
    pub v: RefCell<Tensor>,
}

impl KVCache {
    pub fn new(n_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        // [MaxSeq, Hidden] (Batch=1)
        let hidden = n_heads * head_dim;
        let shape = Shape::new(vec![max_seq_len, hidden]);
        Self {
            k: RefCell::new(Tensor::zeros(shape.clone())),
            v: RefCell::new(Tensor::zeros(shape)),
        }
    }
}

pub struct LlamaAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub n_heads: usize,
    pub cache: KVCache,
}

impl LlamaAttention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        n_heads: usize,
        cache: KVCache,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            cache,
        }
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Tensor {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // 1. RoPE
        let q_r = q.apply_rope(start_pos);
        let k_r = k.apply_rope(start_pos);

        // 2. Cache Update
        let seq_len = x.shape().dims()[0];
        let dim = x.shape().dims()[1];

        {
            let mut c_k = self.cache.k.borrow_mut();
            let mut c_v = self.cache.v.borrow_mut();

            // 단순 메모리 복사 (Native Memory copy logic)
            // (실제로는 Tensor 내부에 copy_from 메서드를 만드는게 좋음)
            let offset = start_pos * dim;
            let len = seq_len * dim;

            let k_data = c_k.data_mut();
            let v_data = c_v.data_mut();

            k_data[offset..offset + len].copy_from_slice(k_r.data());
            v_data[offset..offset + len].copy_from_slice(v.data());
        }

        // 3. Attention (With Cache)
        let c_k = self.cache.k.borrow();
        let c_v = self.cache.v.borrow();

        let total_len = start_pos + seq_len;
        // View 생성 (복사 없이 Shape만 바꾼 Tensor를 만들면 좋겠지만, MVP는 Slice로 새 Tensor 생성)
        // 성능을 위해 전체 Cache 중 유효한 부분만 잘라냅니다.
        let k_view = Tensor::new(
            c_k.data()[0..total_len * dim].to_vec(),
            Shape::new(vec![total_len, dim]),
        );
        let v_view = Tensor::new(
            c_v.data()[0..total_len * dim].to_vec(),
            Shape::new(vec![total_len, dim]),
        );

        // Q @ K.T
        // Q: [Seq, Dim], K: [Total, Dim] -> K.T: [Dim, Total]
        // Score: [Seq, Total]
        let k_t = k_view.transpose();
        let scores = q_r.matmul(&k_t);

        // Scaling
        let head_dim = dim / self.n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scores_scaled = scores.scale(scale); // Tensor::scale 구현 필요

        // Masking (Prompt Phase only)
        if seq_len > 1 {
            // Apply causal mask logic (implement in Tensor)
            // MVP: 생략하거나 Tensor::apply_mask 구현
        }

        let probs = scores_scaled.softmax();

        // Probs @ V
        let context = probs.matmul(&v_view);
        self.o_proj.forward(&context)
    }
}

pub struct LlamaRMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl LlamaRMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.rms_norm(&self.weight, self.eps)
    }
}

pub struct LlamaBlock {
    pub attn: LlamaAttention,
    pub mlp: LlamaMLP,
    pub input_norm: LlamaRMSNorm,
    pub post_norm: LlamaRMSNorm,
}

impl LlamaBlock {
    pub fn forward(&self, x: &Tensor, start_pos: usize) -> Tensor {
        // 1. Attention Block
        // h = x + Attention(Norm(x))
        let normalized_x = self.input_norm.forward(x);
        let attn_out = self.attn.forward(&normalized_x, start_pos);
        let h = x.add(&attn_out);

        // 2. MLP Block
        // out = h + MLP(Norm(h))
        let normalized_h = self.post_norm.forward(&h);
        let mlp_out = self.mlp.forward(&normalized_h);
        let out = h.add(&mlp_out);

        out
    }
}

mod tests {
    use crate::core::shape::Shape;

    use super::*;

    #[test]
    fn test_linear_layer() {
        // 1. 입력 (Batch=1, Input_Dim=2)
        // 값: [1.0, 2.0]
        let input = Tensor::new(vec![1.0, 2.0], Shape::new(vec![1, 2]));

        // 2. 가중치 (Input_Dim=2, Output_Dim=3)
        // [0.5, 1.0, 0.0]
        // [0.5, 0.0, 1.0]
        // 수학적으로 (1x2) @ (2x3) -> (1x3) 행렬 곱셈이 됨
        let weight_data = vec![0.5, 1.0, 0.0, 0.5, 0.0, 1.0];
        let weight = Tensor::new(weight_data, Shape::new(vec![2, 3]));

        // 3. Linear 레이어 생성
        let linear = Linear::new(weight);

        // 4. Forward 실행
        let output = linear.forward(&input);

        // 5. 검증
        // 계산:
        // Out[0] = 1.0*0.5 + 2.0*0.5 = 1.5
        // Out[1] = 1.0*1.0 + 2.0*0.0 = 1.0
        // Out[2] = 1.0*0.0 + 2.0*1.0 = 2.0
        assert_eq!(output.shape().dims(), &[1, 3]); // 차원이 2->3으로 변했는지 확인
        assert_eq!(output.data(), &[1.5, 1.0, 2.0]);
    }

    #[test]
    fn test_swiglu_mlp() {
        // 1. 입력 X (1x2)
        // 값: [1.0, 1.0]
        let input = Tensor::new(vec![1.0, 1.0], Shape::new(vec![1, 2]));

        // 2. 가중치 설정 (계산 검증을 위해 간단한 값 사용)

        // Gate Proj (2x2): Identity Matrix
        // X @ Gate = [1.0, 1.0]
        // Silu([1.0, 1.0]) = [0.731..., 0.731...]
        let w_gate = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));

        // Up Proj (2x2): 2 * Identity Matrix
        // X @ Up = [2.0, 2.0]
        let w_up = Tensor::new(vec![2.0, 0.0, 0.0, 2.0], Shape::new(vec![2, 2]));

        // Down Proj (2x2): Identity Matrix
        let w_down = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));

        // 3. MLP 생성
        let mlp = LlamaMLP {
            gate_proj: Linear::new(w_gate),
            up_proj: Linear::new(w_up),
            down_proj: Linear::new(w_down),
        };

        // 4. Forward
        let output = mlp.forward(&input);

        // 5. 검증
        // Fused = Silu(Gate) * Up
        //       = [0.731058, 0.731058] * [2.0, 2.0]
        //       = [1.462117, 1.462117]
        // Output = Fused @ Down (Identity)
        //        = [1.462117, 1.462117]

        let d = output.data();
        let expected = 1.0 / (1.0 + (-1.0f32).exp()) * 2.0; // 1.462117...

        assert_eq!(output.shape().dims(), &[1, 2]);
        assert!((d[0] - expected).abs() < 0.001);
        assert!((d[1] - expected).abs() < 0.001);
    }

    #[test]
    fn test_attention_forward() {
        // 간단한 설정: Hidden=4, Head=2 (HeadDim=2)
        // Input: [1, 4] (Batch=1, Seq=1, Hidden=4)
        let input = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], Shape::new(vec![1, 4]));

        // 가중치는 모두 Identity로 가정하여 계산 단순화 (MVP Test)
        // Q, K, V, O 모두 Identity
        let w_identity = Tensor::new(
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            Shape::new(vec![4, 4]),
        );

        let attn = LlamaAttention {
            q_proj: Linear::new(w_identity.clone()),
            k_proj: Linear::new(w_identity.clone()),
            v_proj: Linear::new(w_identity.clone()),
            o_proj: Linear::new(w_identity.clone()),
            n_heads: 2,
            cache: KVCache::new(2, 2, 512), // HeadDim=2
        };

        // Forward 실행 (start_pos = 0)
        let output = attn.forward(&input, 0);

        // Shape 확인: [1, 4]
        assert_eq!(output.shape().dims(), &[1, 4]);

        // 값 확인:
        // Q=K=V=Input. RoPE는 pos=0이라 영향 없음.
        // Score = Q @ K.T (유사도) -> Softmax -> V와 곱함.
        // 자기 자신과의 Attention이므로 값이 보존되거나 증폭됨.
        // 적어도 0.0은 아니어야 함.
        assert!(output.data()[0] > 0.0);
    }

    #[test]
    fn test_rms_norm_layer() {
        let input = Tensor::new(vec![1.0, 1.0], Shape::new(vec![1, 2]));
        // Weight = [0.5, 0.5]
        let weight = Tensor::new(vec![0.5, 0.5], Shape::new(vec![2]));

        let norm = LlamaRMSNorm::new(weight, 1e-5);
        let output = norm.forward(&input);

        // RMS([1,1]) = 1.0 -> Norm = [1,1] -> * Weight([0.5,0.5]) = [0.5, 0.5]
        let output_data = output.data();
        let expected = [0.5, 0.5];
        let epsilon = 1e-5; // 허용 오차 범위

        for (i, (&out, &exp)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (out - exp).abs() < epsilon,
                "Index {}: value {} is not close enough to expected {} (diff: {})",
                i,
                out,
                exp,
                (out - exp).abs()
            );
        }
    }

    #[test]
    fn test_llama_block_forward() {
        // Hidden=4, Head=2
        let dim = 4;
        let input = Tensor::new(vec![1.0; dim], Shape::new(vec![1, dim]));

        // 모든 가중치를 Identity나 1.0으로 초기화하여 계산 단순화 (Mocking)
        // 실제로는 Random Init이나 Load를 써야 하지만, 구조 검증이 목적
        let w_identity = Tensor::new(
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            Shape::new(vec![4, 4]),
        );
        let w_ones = Tensor::new(vec![1.0; 4], Shape::new(vec![4])); // Norm weight

        // 부품 조립
        let attn = LlamaAttention {
            q_proj: Linear::new(w_identity.clone()),
            k_proj: Linear::new(w_identity.clone()),
            v_proj: Linear::new(w_identity.clone()),
            o_proj: Linear::new(w_identity.clone()),
            n_heads: 2,
            cache: KVCache::new(2, dim / 2, 512),
        };

        // MLP (SwiGLU) - 차원 변화 없이 테스트 (Intermediate=4)
        let mlp = LlamaMLP::new(
            Linear::new(w_identity.clone()), // Gate
            Linear::new(w_identity.clone()), // Up
            Linear::new(w_identity.clone()), // Down
        );

        let norm = LlamaRMSNorm::new(w_ones.clone(), 1e-5);

        let block = LlamaBlock {
            attn,
            mlp,
            input_norm: norm.new_with(w_ones.clone()), // Helper needed or clone manual
            post_norm: LlamaRMSNorm::new(w_ones.clone(), 1e-5),
        };

        // Forward (start_pos = 0)
        let output = block.forward(&input, 0);

        // 차원이 유지되는지 확인
        assert_eq!(output.shape().dims(), &[1, 4]);
        // 값이 변했는지 확인 (Residual connection 덕분에 0이 아님)
        assert!(output.data()[0] != 0.0);
    }

    // Helper trait to clone LlamaRMSNorm logic for test brevity
    impl LlamaRMSNorm {
        pub fn new_with(&self, w: Tensor) -> Self {
            Self {
                weight: w,
                eps: self.eps,
            }
        }
    }
}
