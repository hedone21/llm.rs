use super::shape::Shape;

#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        if data.len() != shape.num_elements() {
            panic!(
                "Data size mismatch: data len = {}, shape size = {}",
                data.len(),
                shape.num_elements()
            );
        }
        Self { data, shape }
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

    // 헬퍼: 0으로 초기화된 텐서
    pub fn zeros(shape: Shape) -> Self {
        let count = shape.num_elements();
        Self {
            data: vec![0.0; count],
            shape,
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        if self.shape != other.shape {
            panic!("Shape mismatch: {:?} vs {:?}", self.shape, other.shape);
        }

        let result: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Tensor::new(result, self.shape.clone())
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let dims_a = self.shape.dims();
        let dims_b = other.shape.dims();

        // 1. 차원 검증
        if dims_a.len() != 2 || dims_b.len() != 2 {
            panic!("MatMul only supports 2D tensors for now");
        }

        let (m, k) = (dims_a[0], dims_a[1]);
        let (k2, n) = (dims_b[0], dims_b[1]);

        if k != k2 {
            panic!("Dimension mismatch: A({}x{}) @ B({}x{})", m, k, k2, n);
        }

        // 2. 결과 텐서 초기화
        let mut result_data = vec![0.0; m * n];

        // 3. 연산 수행 (Naive Implementation)
        let a_data = &self.data;
        let b_data = &other.data;

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    // A[i][p] * B[p][j]
                    let a_val = a_data[i * k + p];
                    let b_val = b_data[p * n + j];
                    sum += a_val * b_val;
                }
                result_data[i * n + j] = sum;
            }
        }

        Tensor::new(result_data, Shape::new(vec![m, n]))
    }

    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        // 마지막 차원을 기준으로 정규화 수행
        let dims = self.shape.dims();
        let last_dim = *dims.last().expect("Tensor must have at least 1 dim");

        if weight.shape.num_elements() != last_dim {
            panic!("RMSNorm weight shape mismatch");
        }

        let mut output_data = vec![0.0; self.data.len()];
        let w_data = weight.data();

        // 행(Row) 단위로 순회
        for (i, row) in self.data.chunks(last_dim).enumerate() {
            // 1. Calculate Sum of Squares
            let ss: f32 = row.iter().map(|v| v * v).sum();

            // 2. Calculate RMS
            let rms = (ss / last_dim as f32 + eps).sqrt();
            let scale = 1.0 / rms;

            // 3. Normalize & Scale by Weight
            for j in 0..last_dim {
                output_data[i * last_dim + j] = row[j] * scale * w_data[j];
            }
        }

        Tensor::new(output_data, self.shape.clone())
    }

    pub fn silu(&self) -> Tensor {
        let res: Vec<f32> = self.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        Tensor::new(res, self.shape.clone())
    }

    pub fn softmax(&self) -> Tensor {
        let dims = self.shape.dims();
        let last_dim = *dims.last().expect("Tensor rank >= 1");

        let mut output_data = vec![0.0; self.data.len()];

        for (i, row) in self.data.chunks(last_dim).enumerate() {
            // Numerical Stability를 위해 Max 값을 뺌
            let max_val = row.iter().fold(f32::MIN, |a, &b| a.max(b));

            let mut sum_exp = 0.0;
            let mut exps = vec![0.0; last_dim];

            for j in 0..last_dim {
                let e = (row[j] - max_val).exp();
                exps[j] = e;
                sum_exp += e;
            }

            for j in 0..last_dim {
                output_data[i * last_dim + j] = exps[j] / sum_exp;
            }
        }

        Tensor::new(output_data, self.shape.clone())
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        if self.shape != other.shape {
            panic!("Shape mismatch for mul");
        }
        let res: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(res, self.shape.clone())
    }

    // 2D 전치 행렬 (Transpose)
    pub fn transpose(&self) -> Tensor {
        let dims = self.shape.dims();
        if dims.len() != 2 {
            panic!("Transpose only supports 2D for MVP");
        }
        let (m, n) = (dims[0], dims[1]);

        let mut new_data = vec![0.0; self.data.len()];
        for i in 0..m {
            for j in 0..n {
                // Input(i, j) -> Output(j, i)
                new_data[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor::new(new_data, Shape::new(vec![n, m]))
    }

    // RoPE 적용 (간소화 버전)
    // x_new = x * cos(theta) - y * sin(theta)
    // y_new = x * sin(theta) + y * cos(theta)
    pub fn apply_rope(&self, pos: usize) -> Tensor {
        let dims = self.shape.dims();
        let head_dim = *dims.last().unwrap();

        if head_dim % 2 != 0 {
            panic!("Head dimension must be even for RoPE");
        }

        let mut new_data = self.data.clone();

        // 청크 단위(Head Dim)로 순회
        for chunk in new_data.chunks_mut(head_dim) {
            for i in (0..head_dim).step_by(2) {
                // 주파수 계산 (Llama 3 표준: 500,000.0)
                let freq_idx = (i as f32) / (head_dim as f32);
                let theta_base = 500_000.0f32;
                let theta = 1.0 / theta_base.powf(freq_idx);
                let m_theta = (pos as f32) * theta;

                let cos = m_theta.cos();
                let sin = m_theta.sin();

                let x = chunk[i];
                let y = chunk[i + 1];

                // 회전 적용
                chunk[i] = x * cos - y * sin;
                chunk[i + 1] = x * sin + y * cos;
            }
        }

        Tensor::new(new_data, self.shape.clone())
    }

    /// 텐서의 모든 요소에 스칼라 값(value)을 곱합니다.
    /// 용도: Attention Score Scaling (x * 1/sqrt(head_dim))
    pub fn scale(&self, value: f32) -> Tensor {
        let new_data: Vec<f32> = self.data.iter().map(|&x| x * value).collect();

        Tensor::new(new_data, self.shape.clone())
    }
}

mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "Data size mismatch")]
    fn test_shape_mismatch() {
        // 데이터는 3개인데 Shape은 2x2=4개인 경우 패닉이 나야 함
        let shape = Shape::new(vec![2, 2]);
        let data = vec![1.0, 2.0, 3.0];
        Tensor::new(data, shape);
    }

    #[test]
    fn test_create_tensor() {
        // 1. Shape 정의 (2행 3열)
        let shape = Shape::new(vec![2, 3]);

        // 2. 데이터 정의
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // 3. 텐서 생성
        let tensor = Tensor::new(data.clone(), shape.clone());

        // 4. 검증
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data(), &data);
        assert_eq!(tensor.shape().num_elements(), 6);
        assert_eq!(tensor.shape().rank(), 2);
    }

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Tensor::new(vec![0.5, 0.5, 0.5], Shape::new(vec![3]));

        let c = a.add(&b);

        assert_eq!(c.data(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_add_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], Shape::new(vec![2]));
        let b = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        a.add(&b);
    }

    #[test]
    fn test_matmul_basic() {
        // A: (2, 3)
        // [1, 2, 3]
        // [4, 5, 6]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));

        // B: (3, 2)
        // [1, 2]
        // [3, 4]
        // [5, 6]
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![3, 2]));

        // C = A @ B -> (2, 2)
        // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        let c = a.matmul(&b);

        assert_eq!(c.shape().dims(), &[2, 2]);
        assert_eq!(c.data(), &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_matmul_dim_mismatch() {
        let a = Tensor::new(vec![1.0, 1.0], Shape::new(vec![1, 2]));
        let b = Tensor::new(vec![1.0, 1.0, 1.0], Shape::new(vec![3, 1])); // K가 2 != 3
        a.matmul(&b);
    }

    #[test]
    fn test_rms_norm() {
        // Shape: (2, 2)
        // Row 1: [3.0, 4.0] -> 3^2+4^2=25 -> mean=12.5 -> rms=3.5355
        // Row 2: [6.0, 8.0] -> 6^2+8^2=100 -> mean=50 -> rms=7.0710
        let x = Tensor::new(vec![3.0, 4.0, 6.0, 8.0], Shape::new(vec![2, 2]));

        // Weight: [1.0, 1.0] (크기 변화 없이 정규화만 확인)
        let w = Tensor::new(vec![1.0, 1.0], Shape::new(vec![2]));

        let eps = 1e-5;
        let out = x.rms_norm(&w, eps);

        // Row 1 Check: 3/3.5355 ≈ 0.8485, 4/3.5355 ≈ 1.1313
        let d = out.data();
        assert!((d[0] - 0.8485).abs() < 0.001);
        assert!((d[1] - 1.1313).abs() < 0.001);
    }

    #[test]
    fn test_silu() {
        let x = Tensor::new(vec![0.0, 1.0, -1.0], Shape::new(vec![3]));
        let y = x.silu();

        // silu(0) = 0 / (1+1) = 0
        // silu(1) = 1 / (1 + exp(-1)) ≈ 0.731
        let d = y.data();
        assert_eq!(d[0], 0.0);
        assert!((d[1] - 0.731).abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        // [10.0, 10.0] -> 둘 다 같으므로 확률은 0.5, 0.5
        let x = Tensor::new(vec![10.0, 10.0], Shape::new(vec![1, 2]));
        let y = x.softmax();

        assert_eq!(y.data()[0], 0.5);
        assert_eq!(y.data()[1], 0.5);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::new(vec![2.0, 3.0], Shape::new(vec![2]));
        let b = Tensor::new(vec![4.0, 5.0], Shape::new(vec![2]));
        let c = a.mul(&b);
        assert_eq!(c.data(), &[8.0, 15.0]);
    }

    #[test]
    fn test_transpose() {
        // 2x3 행렬
        // [1, 2, 3]
        // [4, 5, 6]
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));

        // Transpose -> 3x2 행렬
        // [1, 4]
        // [2, 5]
        // [3, 6]
        let t_t = t.transpose();

        assert_eq!(t_t.shape().dims(), &[3, 2]);
        assert_eq!(t_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_rope_rotation() {
        // 간단한 RoPE 테스트 (Head Dim = 2)
        // 위치 0: 회전 없음 (cos=1, sin=0) -> 그대로
        // 위치 1: 회전 있음
        let input = Tensor::new(vec![1.0, 0.0], Shape::new(vec![1, 2])); // (1, 2)

        // pos=0일 때
        let out0 = input.apply_rope(0);
        assert_eq!(out0.data(), &[1.0, 0.0]); // 변화 없음

        // pos가 있을 때 (값이 변했는지만 확인, 정확한 수식 검증은 구현부에서)
        let out1 = input.apply_rope(1);
        assert_ne!(out1.data(), &[1.0, 0.0]); // 회전했으므로 달라야 함
    }

    #[test]
    fn test_scale() {
        // 1. 입력 텐서 준비 [2.0, 4.0, 8.0]
        let input = Tensor::new(vec![2.0, 4.0, 8.0], Shape::new(vec![3]));

        // 2. 0.5배 스케일링 수행
        let output = input.scale(0.5);

        // 3. 검증: [1.0, 2.0, 4.0]이 되어야 함
        assert_eq!(output.data(), &[1.0, 2.0, 4.0]);
        assert_eq!(output.shape().dims(), &[3]); // 형태는 유지되어야 함
    }
}
