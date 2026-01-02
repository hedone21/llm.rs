use crate::backend::cpu::CpuBackend;
use crate::backend::{Backend, Device};
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use rand::Rng;

#[cfg(feature = "opencl")]
use crate::backend::opencl::{init_scratch_pool, reset_scratch_pool};

// =========================================================================
// Helper Functions
// =========================================================================

fn generate_random_tensor(shape: Shape) -> Tensor {
    let mut rng = rand::rng();
    let count = shape.num_elements();
    let data: Vec<f32> = (0..count).map(|_| rng.random_range(-1.0..1.0)).collect();
    Tensor::new(data, shape)
}

fn assert_tensor_approx_eq(a: &Tensor, b: &Tensor, epsilon: f32) {
    let a_data = a.data();
    let b_data = b.data();

    if a_data.len() != b_data.len() {
        panic!(
            "FAIL: Tensor size mismatch: {} vs {}",
            a_data.len(),
            b_data.len()
        );
    }

    for i in 0..a_data.len() {
        let diff = (a_data[i] - b_data[i]).abs();
        if diff > epsilon {
            panic!(
                "FAIL: Mismatch at index {}: {} vs {} (diff: {})",
                i, a_data[i], b_data[i], diff
            );
        }
    }
}

// 검증용 Reference Implementation (Naive)
fn reference_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let dim_a = a.shape().dims();
    let dim_b = b.shape().dims();
    let m = dim_a[0];
    let k = dim_a[1];
    let n = dim_b[1]; // B is [K, N]

    let a_data = a.data();
    let b_data = b.data();
    let mut c_data = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
    Tensor::new(c_data, Shape::new(vec![m, n]))
}

fn reference_matmul_transposed(a: &Tensor, b: &Tensor) -> Tensor {
    let dim_a = a.shape().dims();
    let dim_b = b.shape().dims();
    let m = dim_a[0];
    let k = dim_a[1];
    let n = dim_b[0]; // B is [N, K]

    let a_data = a.data();
    let b_data = b.data();
    let mut c_data = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[j * k + p]; // B accessed as [j, p]
            }
            c_data[i * n + j] = sum;
        }
    }
    Tensor::new(c_data, Shape::new(vec![m, n]))
}

// =========================================================================
// Test Runners (Public)
// =========================================================================

pub fn test_device_info() {
    print!("test_device_info ... ");
    let backend = CpuBackend;
    assert_eq!(backend.device(), Device::Cpu);
    assert!(!backend.name().is_empty());
    println!("ok");
}

pub fn test_matmul_f32_correctness() {
    print!("test_matmul_f32_correctness ... ");
    let backend = CpuBackend;

    // Case 1: Decoding (Vector-Matrix)
    let a = generate_random_tensor(Shape::new(vec![1, 128]));
    let b = generate_random_tensor(Shape::new(vec![128, 256]));
    let expected = reference_matmul(&a, &b);
    let result = backend.matmul(&a, &b);
    assert_tensor_approx_eq(&result, &expected, 1e-4);

    // Case 2: Prefill (Matrix-Matrix)
    let a = generate_random_tensor(Shape::new(vec![16, 64]));
    let b = generate_random_tensor(Shape::new(vec![64, 32]));
    let expected = reference_matmul(&a, &b);
    let result = backend.matmul(&a, &b);
    assert_tensor_approx_eq(&result, &expected, 1e-4);

    println!("ok");
}

pub fn test_matmul_transposed_correctness() {
    print!("test_matmul_transposed_correctness ... ");
    let backend = CpuBackend;

    let m = 4;
    let k = 128;
    let n = 64;

    let input = generate_random_tensor(Shape::new(vec![m, k]));
    let weight = generate_random_tensor(Shape::new(vec![n, k]));

    let expected = reference_matmul_transposed(&input, &weight);
    let result = backend.matmul_transposed(&input, &weight);

    assert_tensor_approx_eq(&result, &expected, 1e-4);
    println!("ok");
}

pub fn test_matmul_slice_zero_copy() {
    print!("test_matmul_slice_zero_copy ... ");
    let backend = CpuBackend;

    // Q: [1, 64]
    // K (Cache): [10, 64] -> Treat as if it's transposed during calculation
    let q = generate_random_tensor(Shape::new(vec![1, 64]));
    let k_cache = generate_random_tensor(Shape::new(vec![10, 64]));

    let slice_data = k_cache.data();
    let rows = 10;
    let cols = 64;

    let result = backend.matmul_slice(&q, slice_data, rows, cols);
    let expected = reference_matmul_transposed(&q, &k_cache);

    assert_tensor_approx_eq(&result, &expected, 1e-4);
    println!("ok");
}

pub fn test_elementwise_ops() {
    print!("test_elementwise_ops ... ");
    let backend = CpuBackend;

    // Add Assign
    let mut a = generate_random_tensor(Shape::new(vec![10, 10]));
    let b = generate_random_tensor(Shape::new(vec![10, 10]));
    let a_orig = Tensor::new(a.data().to_vec(), a.shape().clone());
    backend.add_assign(&mut a, &b);

    let a_d = a.data();
    let ao_d = a_orig.data();
    let b_d = b.data();
    for i in 0..a_d.len() {
        assert!((a_d[i] - (ao_d[i] + b_d[i])).abs() < 1e-5);
    }

    // SiLU Mul
    let mut gate = generate_random_tensor(Shape::new(vec![5, 5]));
    let up = generate_random_tensor(Shape::new(vec![5, 5]));
    let g_orig = Tensor::new(gate.data().to_vec(), gate.shape().clone());
    backend.silu_mul(&mut gate, &up);

    let res_d = gate.data();
    let go_d = g_orig.data();
    let u_d = up.data();
    for i in 0..res_d.len() {
        let x = go_d[i];
        let silu = x / (1.0 + (-x).exp());
        assert!((res_d[i] - (silu * u_d[i])).abs() < 1e-5);
    }

    println!("ok");
}

pub fn test_rms_norm() {
    print!("test_rms_norm ... ");
    let backend = CpuBackend;
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![1, 4]));
    let w = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![4]));
    let eps = 1e-5;

    let result = backend.rms_norm(&x, &w, eps);
    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) approx 2.7386
    // [0.365, 0.730, 1.095, 1.460]

    let d = result.data();
    assert!((d[0] - 0.3651).abs() < 1e-3);
    assert!((d[3] - 1.4605).abs() < 1e-3);
    println!("ok");
}

pub fn test_softmax() {
    print!("test_softmax ... ");
    let backend = CpuBackend;
    let x = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
    let result = backend.softmax(&x);
    let d = result.data();

    // Check sum = 1.0
    let sum: f32 = d.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Check ordering
    assert!(d[2] > d[1] && d[1] > d[0]);
    println!("ok");
}

pub fn test_rope_inplace() {
    print!("test_rope_inplace ... ");
    let backend = CpuBackend;
    let mut x = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![1, 4]));
    backend.rope_inplace(&mut x, 1);

    let d = x.data();
    // Magnitude check
    let mag1 = (d[0].powi(2) + d[2].powi(2)).sqrt();
    let mag2 = (d[1].powi(2) + d[3].powi(2)).sqrt();
    assert!((mag1 - 1.0).abs() < 1e-5);
    assert!((mag2 - 1.0).abs() < 1e-5);

    // Value change check (rotation)
    assert_ne!(d[0], 1.0);
    println!("ok");
}

// 통합 실행 함수
pub fn run_all_tests() {
    println!("=== Running CPU Backend Tests (Standalone Binary) ===");

    #[cfg(feature = "opencl")]
    {
        // 테스트용으로 128MB 정도만 할당
        init_scratch_pool(128);
    }

    test_device_info();
    test_matmul_f32_correctness();
    #[cfg(feature = "opencl")]
    reset_scratch_pool(); // 연산 사이 리셋
    test_matmul_transposed_correctness();
    #[cfg(feature = "opencl")]
    reset_scratch_pool(); // 연산 사이 리셋
    test_matmul_slice_zero_copy();
    #[cfg(feature = "opencl")]
    reset_scratch_pool(); // 연산 사이 리셋
    test_elementwise_ops();
    #[cfg(feature = "opencl")]
    reset_scratch_pool(); // 연산 사이 리셋
    test_rms_norm();
    #[cfg(feature = "opencl")]
    reset_scratch_pool(); // 연산 사이 리셋
    test_softmax();
    #[cfg(feature = "opencl")]
    reset_scratch_pool(); // 연산 사이 리셋
    test_rope_inplace();
    println!("=== All CPU Tests Passed! ===");
}
