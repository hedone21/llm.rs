use super::shape::Shape;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Tensor {
    name: Option<String>,
    pub data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        if data.len() != shape.num_elements() {
            panic!("Data size mismatch");
        }
        Self {
            name: None,
            data,
            shape,
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn zeros(shape: Shape) -> Self {
        let count = shape.num_elements();
        Self {
            name: None,
            data: vec![0.0; count],
            shape,
        }
    }

    pub fn add_assign(&mut self, other: &Tensor) {
        if self.shape != other.shape {
            panic!("Shape mismatch in add_assign");
        }
        self.data
            .par_iter_mut()
            .zip(other.data.par_iter())
            .for_each(|(a, b)| *a += *b);
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let mut res = self.clone();
        res.add_assign(other);
        res
    }

    pub fn silu_mul_inplace(&mut self, other: &Tensor) {
        if self.shape != other.shape {
            panic!("Shape mismatch in silu_mul_inplace");
        }
        self.data
            .par_iter_mut()
            .zip(other.data.par_iter())
            .for_each(|(x, y)| {
                let val = *x;
                let silu = val / (1.0 + (-val).exp());
                *x = silu * *y;
            });
    }

    // [수정] 표준 행렬 곱 (A @ B) - 올바른 로직으로 복구
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let dims_a = self.shape.dims();
        let dims_b = other.shape.dims();

        if dims_a.len() != 2 || dims_b.len() != 2 {
            panic!("MatMul only supports 2D tensors");
        }
        let (m, k) = (dims_a[0], dims_a[1]);
        let (k2, n) = (dims_b[0], dims_b[1]);

        if k != k2 {
            panic!(
                "MatMul Dimension mismatch: ({}, {}) @ ({}, {})",
                m, k, k2, n
            );
        }

        let mut result_data = vec![0.0; m * n];
        let a_data = &self.data;
        let b_data = &other.data;

        // 병렬 처리 (B가 연속적이지 않으므로 cache 효율은 낮지만 로직은 정확함)
        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, res_row)| {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        unsafe {
                            sum +=
                                *a_data.get_unchecked(i * k + p) * *b_data.get_unchecked(p * n + j);
                        }
                    }
                    res_row[j] = sum;
                }
            });

        Tensor::new(result_data, Shape::new(vec![m, n]))
    }

    // [최적화] A @ B^T (B가 이미 Transposed된 상태일 때 사용)
    pub fn matmul_transposed(&self, other_t: &Tensor) -> Tensor {
        let dims_a = self.shape.dims();
        let dims_b_t = other_t.shape.dims();
        let (m, k) = (dims_a[0], dims_a[1]);
        let (n, k2) = (dims_b_t[0], dims_b_t[1]);

        if k != k2 {
            panic!(
                "Dimension mismatch for matmul_transposed: A[{},{}] vs B_t[{},{}]",
                m, k, n, k2
            );
        }

        let mut result_data = vec![0.0; m * n];
        let a_data = &self.data;
        let b_data = &other_t.data;

        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, res_row)| {
                let a_row_start = i * k;
                let a_slice = &a_data[a_row_start..a_row_start + k];
                for j in 0..n {
                    let b_row_start = j * k;
                    let b_slice = &b_data[b_row_start..b_row_start + k];
                    let sum: f32 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum();
                    res_row[j] = sum;
                }
            });
        Tensor::new(result_data, Shape::new(vec![m, n]))
    }

    pub fn transpose(&self) -> Tensor {
        let dims = self.shape.dims();
        let (m, n) = (dims[0], dims[1]);
        let mut new_data = vec![0.0; self.data.len()];
        for i in 0..m {
            for j in 0..n {
                new_data[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor::new(new_data, Shape::new(vec![n, m]))
    }

    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        let dims = self.shape.dims();
        let last_dim = *dims.last().unwrap();
        let mut output_data = vec![0.0; self.data.len()];
        let w_data = weight.data();

        output_data
            .par_chunks_mut(last_dim)
            .zip(self.data.par_chunks(last_dim))
            .for_each(|(out_row, in_row)| {
                let ss: f32 = in_row.iter().map(|v| v * v).sum();
                let rms = (ss / last_dim as f32 + eps).sqrt();
                let scale = 1.0 / rms;
                for j in 0..last_dim {
                    out_row[j] = in_row[j] * scale * w_data[j];
                }
            });
        Tensor::new(output_data, self.shape.clone())
    }

    pub fn softmax(&self) -> Tensor {
        let dims = self.shape.dims();
        let last_dim = *dims.last().unwrap();
        let mut output_data = vec![0.0; self.data.len()];

        output_data
            .par_chunks_mut(last_dim)
            .zip(self.data.par_chunks(last_dim))
            .for_each(|(out_row, in_row)| {
                let max_val = in_row.iter().fold(f32::MIN, |a, &b| a.max(b));
                let mut sum_exp = 0.0;
                for j in 0..last_dim {
                    let e = (in_row[j] - max_val).exp();
                    out_row[j] = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                for j in 0..last_dim {
                    out_row[j] *= inv_sum;
                }
            });
        Tensor::new(output_data, self.shape.clone())
    }

    pub fn scale(&self, value: f32) -> Tensor {
        let new_data: Vec<f32> = self.data.par_iter().map(|&x| x * value).collect();
        Tensor::new(new_data, self.shape.clone())
    }

    pub fn apply_rope_inplace(&mut self, start_pos: usize) {
        let dims = self.shape.dims();
        let head_dim = *dims.last().unwrap();
        let mid = head_dim / 2;
        let theta_base = 500_000.0f32;

        self.data
            .par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(i, chunk)| {
                let pos = start_pos + i;
                for j in 0..mid {
                    let freq_idx = (j as f32 * 2.0) / (head_dim as f32);
                    let theta = 1.0 / theta_base.powf(freq_idx);
                    let m_theta = (pos as f32) * theta;
                    let (sin, cos) = m_theta.sin_cos();

                    let x = chunk[j];
                    let y = chunk[j + mid];

                    chunk[j] = x * cos - y * sin;
                    chunk[j + mid] = x * sin + y * cos;
                }
            });
    }

    pub fn apply_rope(&self, pos: usize) -> Tensor {
        let mut res = self.clone();
        res.apply_rope_inplace(pos);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. MatMul 테스트: (1x2) @ (2x2) 행렬 곱 확인
    #[test]
    fn test_matmul_simple() {
        let a = Tensor::new(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        // Identity Matrix
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));

        let c = a.matmul(&b);
        assert_eq!(c.data(), &[1.0, 2.0]);
    }

    // 2. RoPE 테스트 (핵심!)
    #[test]
    fn test_rope_correctness() {
        // HeadDim = 2 (짝수여야 함)
        let input = Tensor::new(vec![1.0, 0.0], Shape::new(vec![1, 2]));

        // Case A: Position 0
        // cos(0)=1, sin(0)=0 이므로 값의 변화가 없어야 함
        let out_0 = input.apply_rope(0);
        assert_eq!(out_0.data(), &[1.0, 0.0], "RoPE at pos 0 must be Identity");

        // Case B: Position > 0
        // 값이 변해야 함 (회전 발생)
        let out_1 = input.apply_rope(1);
        assert_ne!(out_1.data(), &[1.0, 0.0], "RoPE at pos 1 must rotate");

        // 수동 검증: Theta Base = 500,000
        // i=0 (첫번째 쌍), freq_idx = 0/2 = 0 -> theta = 1.0
        // pos=1 -> m_theta = 1.0
        // x=1, y=0
        // out[0] = 1*cos(1) - 0*sin(1) = cos(1) ≈ 0.5403
        // out[1] = 1*sin(1) + 0*cos(1) = sin(1) ≈ 0.8414
        let d = out_1.data();
        assert!((d[0] - 0.5403).abs() < 0.01);
        assert!((d[1] - 0.8414).abs() < 0.01);
    }

    // 3. RMS Norm 테스트
    #[test]
    fn test_rms_norm() {
        // [3.0, 4.0] -> 제곱합 25 -> 평균 12.5 -> sqrt(12.5) ≈ 3.5355
        // Norm -> [3/3.53, 4/3.53] ≈ [0.848, 1.131]
        let input = Tensor::new(vec![3.0, 4.0], Shape::new(vec![1, 2]));
        let weight = Tensor::new(vec![1.0, 1.0], Shape::new(vec![2])); // 가중치 1

        let out = input.rms_norm(&weight, 1e-5);
        let d = out.data();

        assert!((d[0] - 0.8485).abs() < 0.001);
        assert!((d[1] - 1.1313).abs() < 0.001);
    }

    // 4. Softmax 테스트 (확률의 합은 1.0)
    #[test]
    fn test_softmax_sum_to_one() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let out = input.softmax();
        let sum: f32 = out.data().iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);
    }
}
