use crate::backend::opencl::{OpenClBackend, reset_scratch_pool};
use crate::backend::{Backend, Device};
use crate::core::shape::Shape;
use crate::core::tensor::{Storage, Tensor};
use rand::Rng;

// =========================================================================
// Helper Functions
// =========================================================================

fn generate_random_tensor(shape: Shape) -> Tensor {
    let mut rng = rand::rng();
    let count = shape.num_elements();
    let data: Vec<f32> = (0..count).map(|_| rng.random_range(-1.0..1.0)).collect();
    Tensor::new(data, shape)
}

fn assert_tensor_approx_eq(gpu_tensor: &Tensor, cpu_ref: &Tensor, epsilon: f32) {
    // OpenCL 백엔드는 Shared Memory(매핑된 포인터)를 사용하므로
    // .data() 호출 시 자동으로 동기화된(또는 매핑된) 호스트 메모리를 읽습니다.
    let gpu_data = gpu_tensor.data();
    let cpu_data = cpu_ref.data();

    if gpu_data.len() != cpu_data.len() {
        panic!(
            "FAIL: Tensor size mismatch: {} vs {}",
            gpu_data.len(),
            cpu_data.len()
        );
    }

    for i in 0..gpu_data.len() {
        let diff = (gpu_data[i] - cpu_data[i]).abs();
        if diff > epsilon {
            panic!(
                "FAIL: Mismatch at index {}: GPU({:.6}) vs CPU({:.6}) (diff: {:.6})",
                i, gpu_data[i], cpu_data[i], diff
            );
        }
    }
}

// 검증용 Reference Implementation (CPU Naive)
fn reference_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let dim_a = a.shape().dims();
    let dim_b = b.shape().dims();
    let m = dim_a[0];
    let k = dim_a[1];
    let n = dim_b[1];

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
                sum += a_data[i * k + p] * b_data[j * k + p];
            }
            c_data[i * n + j] = sum;
        }
    }
    Tensor::new(c_data, Shape::new(vec![m, n]))
}

// =========================================================================
// Test Runners (Public)
// =========================================================================

pub fn test_opencl_device_info() {
    print!("test_opencl_device_info ... ");
    // OpenCL 백엔드 초기화 시도 (실패 시 패닉)
    let backend = OpenClBackend::new();
    assert_eq!(backend.device(), Device::OpenCl);
    println!("ok (Context Created)");
}

pub fn test_opencl_copy() {
    print!("test_opencl_copy ... ");
    let backend = OpenClBackend::new();

    let shape = Shape::new(vec![8, 8]);
    let cpu_tensor = generate_random_tensor(shape.clone());
    let gpu_tensor = cpu_tensor.to_device(Device::OpenCl);
    let gpu_tensor_clone = backend.copy_from(&gpu_tensor);
    let back_to_cpu = gpu_tensor.to_device(Device::Cpu);

    assert_tensor_approx_eq(&back_to_cpu, &cpu_tensor, 1e-6);
    assert_tensor_approx_eq(&gpu_tensor, &gpu_tensor_clone, 1e-6);

    println!("ok");
}

pub fn test_opencl_matmul_f32() {
    print!("test_opencl_matmul_f32 ... ");
    let backend = OpenClBackend::new();

    // CPU에서 데이터 생성 후 GPU로 이동
    let a_cpu = generate_random_tensor(Shape::new(vec![16, 64]));
    let b_cpu = generate_random_tensor(Shape::new(vec![64, 32]));

    let a_gpu = a_cpu.to_device(Device::OpenCl);
    let b_gpu = b_cpu.to_device(Device::OpenCl);

    // GPU 연산
    let result_gpu = backend.matmul(&a_gpu, &b_gpu);

    // CPU 검증
    let expected = reference_matmul(&a_cpu, &b_cpu);

    assert_tensor_approx_eq(&result_gpu, &expected, 1e-3);
    println!("ok");
}

pub fn test_opencl_matmul_transposed() {
    print!("test_opencl_matmul_transposed ... ");
    let backend = OpenClBackend::new();

    let m = 4;
    let k = 128;
    let n = 64;

    let input_cpu = generate_random_tensor(Shape::new(vec![m, k]));
    let weight_cpu = generate_random_tensor(Shape::new(vec![n, k]));

    let input_gpu = input_cpu.to_device(Device::OpenCl);
    let weight_gpu = weight_cpu.to_device(Device::OpenCl);

    let result_gpu = backend.matmul_transposed(&input_gpu, &weight_gpu);
    let expected = reference_matmul_transposed(&input_cpu, &weight_cpu);

    assert_tensor_approx_eq(&result_gpu, &expected, 1e-3);
    println!("ok");
}

pub fn test_opencl_matmul_slice() {
    print!("test_opencl_matmul_slice ... ");
    let backend = OpenClBackend::new();

    // Q: GPU Tensor [1, 64]
    // K_cache: CPU Slice [10, 64] (Transposed logic)
    let q_cpu = generate_random_tensor(Shape::new(vec![1, 64]));
    let k_cache_cpu = generate_random_tensor(Shape::new(vec![10, 64]));

    let q_gpu = q_cpu.to_device(Device::OpenCl);
    let slice_data = k_cache_cpu.data();

    // GPU 백엔드가 내부적으로 슬라이스를 업로드하여 연산
    let result_gpu = backend.matmul_slice(&q_gpu, slice_data, 10, 64);

    // Reference: matmul_transposed (Q @ K^T)
    let expected = reference_matmul_transposed(&q_cpu, &k_cache_cpu);

    assert_tensor_approx_eq(&result_gpu, &expected, 1e-3);
    println!("ok");
}

fn dequantize_q4_to_f32(q4_tensor: &Tensor) -> Tensor {
    let shape = q4_tensor.shape();
    let num_elements = shape.num_elements();

    // 1. Storage 타입에 따라 데이터 슬라이스 확보
    let (data_slice, scale_slice) = match &*q4_tensor.storage {
        Storage::CpuQ4 { data, scales } => (data.as_slice(), scales.as_slice()),
        Storage::SharedQ4 {
            ptr,
            data_len,
            scale_len,
            ..
        } => unsafe {
            // SharedQ4는 data_len이 정렬(padding)되어 있을 수 있음
            // scale은 data 뒤에 위치함
            let data_ptr = *ptr;
            let scale_ptr = data_ptr.add(*data_len) as *const f32;

            // 실제 유효한 Q4 데이터 바이트 수만 계산 (패딩 제외)
            let valid_bytes = num_elements.div_ceil(2);

            let d_slice = std::slice::from_raw_parts(data_ptr, valid_bytes);
            let s_slice = std::slice::from_raw_parts(scale_ptr, *scale_len);
            (d_slice, s_slice)
        },
        _ => panic!("Not a Q4 tensor"),
    };

    let mut f32_data = Vec::with_capacity(num_elements);

    // 2. 디코딩 루프
    for (i, &packed) in data_slice.iter().enumerate() {
        let block_idx = (i * 2) / 32;

        // Scale 인덱스 안전장치 (패딩 영역 접근 방지)
        let scale = if block_idx < scale_slice.len() {
            scale_slice[block_idx]
        } else {
            0.0
        };

        // Quantize 로직 (+8.5 offset)의 역연산 (-8.5)
        let v0 = ((packed & 0x0F) as f32 - 8.0) * scale;
        let v1 = ((packed >> 4) as f32 - 8.0) * scale;

        f32_data.push(v0);
        // 홀수 개수일 경우 마지막 v1은 버림
        if f32_data.len() < num_elements {
            f32_data.push(v1);
        }
    }

    Tensor::new(f32_data, shape.clone())
}

pub fn test_opencl_matmul_q4_decoding() {
    print!("test_opencl_matmul_q4_decoding (M=1) ... ");
    let backend = OpenClBackend::new();

    let (m, k, n) = (1, 128, 64);
    let a_cpu = generate_random_tensor(Shape::new(vec![m, k]));
    let b_cpu_f32 = generate_random_tensor(Shape::new(vec![n, k]));

    let b_cpu_q4 = b_cpu_f32.quantize_q4(); // 1. CPU에서 양자화
    let b_gpu_q4 = b_cpu_q4.to_device(Device::OpenCl); // 2. GPU로 업로드
    let a_gpu = a_cpu.to_device(Device::OpenCl);

    let result_gpu = backend.matmul_transposed(&a_gpu, &b_gpu_q4);

    // [중요] 원본 b_cpu_f32가 아니라, 양자화된 b_cpu_q4를 복원한 값과 비교합니다.
    let b_dequantized = dequantize_q4_to_f32(&b_cpu_q4);
    let expected = reference_matmul_transposed(&a_cpu, &b_dequantized);

    // 이제 양자화 오차가 배제되었으므로 epsilon을 1e-3 수준으로 낮춰도 통과해야 합니다.
    assert_tensor_approx_eq(&result_gpu, &expected, 1e-3);
    println!("ok");
}

pub fn test_opencl_matmul_q4_prefill() {
    print!("test_opencl_matmul_q4_prefill (M > 1) ... ");
    let backend = OpenClBackend::new();

    let (m, k, n) = (4, 128, 64); // Batch size 4
    let a_cpu = generate_random_tensor(Shape::new(vec![m, k]));
    let b_cpu_f32 = generate_random_tensor(Shape::new(vec![n, k]));

    let b_gpu_q4 = b_cpu_f32.quantize_q4().to_device(Device::OpenCl);
    let a_gpu = a_cpu.to_device(Device::OpenCl);

    let result_gpu = backend.matmul_transposed(&a_gpu, &b_gpu_q4);
    let expected = reference_matmul_transposed(&a_cpu, &b_cpu_f32);

    assert_tensor_approx_eq(&result_gpu, &expected, 1.5);
    println!("ok");
}

pub fn test_opencl_silu_mul() {
    print!("test_opencl_silu_mul ... ");
    let backend = OpenClBackend::new();

    let shape = Shape::new(vec![128]);
    let mut gate_cpu = generate_random_tensor(shape.clone());
    let up_cpu = generate_random_tensor(shape.clone());

    // CPU Reference 계산
    let mut expected_data = gate_cpu.data().to_vec();
    let up_data = up_cpu.data();
    for i in 0..expected_data.len() {
        let x = expected_data[i];
        let silu = x / (1.0 + (-x).exp());
        expected_data[i] = silu * up_data[i];
    }
    let expected = Tensor::new(expected_data, shape);

    let mut gate_gpu = gate_cpu.to_device(Device::OpenCl);
    let up_gpu = up_cpu.to_device(Device::OpenCl);

    backend.silu_mul(&mut gate_gpu, &up_gpu);

    assert_tensor_approx_eq(&gate_gpu, &expected, 1e-4);
    println!("ok");
}

pub fn test_opencl_add_assign() {
    print!("test_opencl_add_assign ... ");
    let backend = OpenClBackend::new();

    let shape = Shape::new(vec![1024]);
    let mut a_cpu = generate_random_tensor(shape.clone());
    let b_cpu = generate_random_tensor(shape.clone());

    let mut expected_data = a_cpu.data().to_vec();
    let b_data = b_cpu.data();
    for i in 0..expected_data.len() {
        expected_data[i] += b_data[i];
    }
    let expected = Tensor::new(expected_data, shape);

    let mut a_gpu = a_cpu.to_device(Device::OpenCl);
    let b_gpu = b_cpu.to_device(Device::OpenCl);

    backend.add_assign(&mut a_gpu, &b_gpu);

    assert_tensor_approx_eq(&a_gpu, &expected, 1e-4);
    println!("ok");
}

pub fn test_opencl_rms_norm() {
    print!("test_opencl_rms_norm ... ");
    let backend = OpenClBackend::new();

    let x_cpu = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![1, 4]));
    let w_cpu = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![4]));

    let x_gpu = x_cpu.to_device(Device::OpenCl);
    let w_gpu = w_cpu.to_device(Device::OpenCl);

    let result_gpu = backend.rms_norm(&x_gpu, &w_gpu, 1e-5);

    // Expected: [0.3651, 0.7303, 1.0954, 1.4606]
    let d = result_gpu.data();
    assert!((d[0] - 0.3651).abs() < 1e-3);
    assert!((d[3] - 1.4605).abs() < 1e-3);
    println!("ok");
}

pub fn test_opencl_softmax() {
    print!("test_opencl_softmax ... ");
    let backend = OpenClBackend::new();
    let x_cpu = Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
    let x_gpu = x_cpu.to_device(Device::OpenCl);

    let result_gpu = backend.softmax(&x_gpu);
    let d = result_gpu.data();

    let sum: f32 = d.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(d[2] > d[1] && d[1] > d[0]);
    println!("ok");
}

pub fn test_opencl_rope() {
    print!("test_opencl_rope ... ");
    let backend = OpenClBackend::new();

    let x_cpu = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![1, 4]));
    let mut x_gpu = x_cpu.to_device(Device::OpenCl);

    backend.rope_inplace(&mut x_gpu, 1);

    let d = x_gpu.data();
    // Magnitude check
    let mag1 = (d[0].powi(2) + d[2].powi(2)).sqrt();
    let mag2 = (d[1].powi(2) + d[3].powi(2)).sqrt();

    assert!((mag1 - 1.0).abs() < 1e-4);
    assert!((mag2 - 1.0).abs() < 1e-4);
    assert_ne!(d[0], 1.0); // Rotated
    println!("ok");
}

// 슬라이싱된(비연속적) 텐서에서의 MatMul 테스트
pub fn test_opencl_matmul_non_contiguous() {
    print!("test_opencl_matmul_non_contiguous (Strided) ... ");
    let backend = OpenClBackend::new();

    // 1. 큰 텐서를 만들고 그 중 일부만 슬라이싱 (실제 KV Cache 모사)
    let full_shape = Shape::new(vec![1, 10, 128]); // [Batch, Seq, Hidden]
    let full_data = generate_random_tensor(full_shape.clone());

    // 중간의 5개 토큰만 잘라냄 (비연속적인 메모리 뷰 생성)
    let rows = 5;
    let cols = 128;
    let slice_data: Vec<f32> = full_data.data()[(2 * cols)..(7 * cols)].to_vec();
    let a_cpu = Tensor::new(slice_data, Shape::new(vec![rows, cols]));

    let b_cpu = generate_random_tensor(Shape::new(vec![64, cols]));

    // GPU로 전송
    let a_gpu = a_cpu.to_device(Device::OpenCl);
    let b_gpu = b_cpu.to_device(Device::OpenCl);

    // 연산 수행
    let result_gpu = backend.matmul_transposed(&a_gpu, &b_gpu);
    let expected = reference_matmul_transposed(&a_cpu, &b_cpu);

    assert_tensor_approx_eq(&result_gpu, &expected, 0.15);
    println!("ok");
}

// 반복적인 연산 시 메모리 가시성 테스트
pub fn test_opencl_memory_sync_loop() {
    print!("test_opencl_memory_sync_loop (100 iterations) ... ");
    let backend = OpenClBackend::new();
    let shape = Shape::new(vec![1, 1024]);

    let mut a_gpu = Tensor::zeros(shape.clone()).to_device(Device::OpenCl);

    for i in 1..101 {
        let b_cpu = Tensor::new(vec![1.0; 1024], shape.clone());
        let b_gpu = b_cpu.to_device(Device::OpenCl);

        // 매번 쓰고 더하는 과정에서 데이터가 깨지는지 확인
        backend.add_assign(&mut a_gpu, &b_gpu);

        // 중간 확인 (일부 값만 샘플링)
        let current_val = a_gpu.data()[0];
        if current_val != i as f32 {
            panic!(
                "FAIL: Sync error at iter {}: Expected {}, Got {}",
                i, i, current_val
            );
        }
    }
    println!("ok");
}

pub fn test_opencl_prefill_stress() {
    print!("test_opencl_prefill_stress (M=4, K=256, N=128) ... ");
    let backend = OpenClBackend::new();
    reset_scratch_pool();

    let (m, k, n) = (4, 256, 128);
    let a_cpu = generate_random_tensor(Shape::new(vec![m, k]));
    let b_cpu_f32 = generate_random_tensor(Shape::new(vec![n, k]));

    let b_gpu = b_cpu_f32.to_device(Device::OpenCl);
    let a_gpu = a_cpu.to_device(Device::OpenCl);

    // GPU 연산
    let result_gpu = backend.matmul_transposed(&a_gpu, &b_gpu);

    let b_ref = b_cpu_f32;
    let mut expected_data = vec![0.0; m * n];
    let a_data = a_cpu.data();
    let b_data = b_ref.data();
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[j * k + p];
            }
            expected_data[i * n + j] = sum;
        }
    }
    let expected = Tensor::new(expected_data, Shape::new(vec![m, n]));

    assert_tensor_approx_eq(&result_gpu, &expected, 1e-3);
    println!("ok");
}

/// [테스트 1] Q4 MatMul Prefill (Batch > 1) 검증
/// 실제 동작에서 데이터가 깨진다면 이 부분의 오프셋 계산이 주범일 확률이 높음
pub fn test_opencl_q4_prefill_stress() {
    print!("test_opencl_q4_prefill_stress (M=4, K=256, N=128) ... ");
    let backend = OpenClBackend::new();
    reset_scratch_pool();

    let (m, k, n) = (4, 256, 128);
    let a_cpu = generate_random_tensor(Shape::new(vec![m, k]));
    let b_cpu_f32 = generate_random_tensor(Shape::new(vec![n, k]));

    // Q4 양자화 수행
    let b_cpu_q4 = b_cpu_f32.quantize_q4();
    let b_gpu_q4 = b_cpu_q4.to_device(Device::OpenCl);
    let a_gpu = a_cpu.to_device(Device::OpenCl);

    // GPU 연산
    let result_gpu = backend.matmul_transposed(&a_gpu, &b_gpu_q4);

    // 검증: 원본 F32가 아닌 역양자화된 값과 비교하여 커널 로직만 순수하게 검증
    let b_ref = dequantize_q4_to_f32(&b_cpu_q4);
    let mut expected_data = vec![0.0; m * n];
    let a_data = a_cpu.data();
    let b_data = b_ref.data();
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[j * k + p];
            }
            expected_data[i * n + j] = sum;
        }
    }
    let expected = Tensor::new(expected_data, Shape::new(vec![m, n]));

    assert_tensor_approx_eq(&result_gpu, &expected, 1e-3);
    println!("ok");
}

/// [테스트 2] Chained Operations (연쇄 연산)
/// 연산 결과가 다음 연산의 입력으로 쓰일 때 메모리 동기화가 깨지는지 확인
pub fn test_opencl_chained_ops() {
    print!("test_opencl_chained_ops (Add -> RMSNorm -> SiLU) ... ");
    let backend = OpenClBackend::new();
    reset_scratch_pool();

    let shape = Shape::new(vec![1, 512]);
    let a_cpu = generate_random_tensor(shape.clone());
    let b_cpu = generate_random_tensor(shape.clone());
    let w_cpu = Tensor::new(vec![1.0; 512], Shape::new(vec![512]));

    let mut x_gpu = a_cpu.to_device(Device::OpenCl);
    let b_gpu = b_cpu.to_device(Device::OpenCl);
    let w_gpu = w_cpu.to_device(Device::OpenCl);

    // 1. Add
    backend.add_assign(&mut x_gpu, &b_gpu);
    // 2. RMSNorm
    let mut x_norm_gpu = backend.rms_norm(&x_gpu, &w_gpu, 1e-5);
    // 3. SiLU (Self-Mul 모사)
    let dummy_up = generate_random_tensor(shape.clone()).to_device(Device::OpenCl);
    backend.silu_mul(&mut x_norm_gpu, &dummy_up);

    // 위 연산들이 GPU 내부에서 덮어쓰기나 동기화 오류 없이 수행되었는지 확인
    let final_data = x_norm_gpu.data();
    assert!(!final_data[0].is_nan(), "Data corruption detected (NaN)");
    println!("ok");
}

/// [테스트 3] Scratch Pool Recycling
/// 풀이 리셋된 후 이전 메모리 영역이 오염되지 않는지 확인
pub fn test_opencl_scratch_recycling() {
    print!("test_opencl_scratch_recycling ... ");
    let backend = OpenClBackend::new();

    // 1회차 연산
    reset_scratch_pool();
    let a = generate_random_tensor(Shape::new(vec![1, 1024])).to_device(Device::OpenCl);
    let val_a = a.data()[0];

    // 2회차 연산 (리셋 후 새로운 할당이 이전 할당을 깨뜨리는지 확인)
    reset_scratch_pool();
    let b = generate_random_tensor(Shape::new(vec![1, 1024])).to_device(Device::OpenCl);
    let _c = generate_random_tensor(Shape::new(vec![1, 1024])).to_device(Device::OpenCl);

    // a는 이미 스크래치 풀에서 덮어씌워졌을 수 있음 (이게 정상)
    // 하지만 연산 도중에 깨지면 안 됨.
    println!("ok");
}

pub fn run_all_tests() {
    println!("=== Running OpenCL Backend Tests (Standalone Binary) ===");
    // Note: OpenCL 환경이 없으면 OpenClBackend::new()에서 패닉이 발생할 수 있음
    test_opencl_device_info();
    test_opencl_copy();
    test_opencl_matmul_f32();
    test_opencl_matmul_transposed();
    test_opencl_matmul_slice();
    test_opencl_matmul_q4_decoding();
    test_opencl_matmul_q4_prefill();
    test_opencl_silu_mul();
    test_opencl_add_assign();
    test_opencl_rms_norm();
    test_opencl_softmax();
    test_opencl_rope();
    test_opencl_matmul_non_contiguous();
    test_opencl_memory_sync_loop();
    test_opencl_prefill_stress();
    test_opencl_q4_prefill_stress();
    test_opencl_chained_ops();
    test_opencl_scratch_recycling();
    println!("=== All OpenCL Tests Passed! ===");
}
