use super::shape::Shape;
use std::sync::Arc;
// backend 모듈과 backend_impl 모듈이 core에 있다고 가정합니다.
use crate::backend::cpu::CpuBackend;
use crate::backend::{Backend, Device};

#[derive(Debug)]
pub enum Storage {
    // 1. 일반 CPU 메모리 (Vec<f32>)
    Cpu(Vec<f32>),

    // 2. 안드로이드 공유 메모리 (Zero-Copy)
    // CPU는 ptr로 접근, GPU/NPU는 handle(fd)로 접근
    Shared {
        ptr: *mut u8,
        handle: usize, // file descriptor or AHardwareBuffer pointer
        size: usize,
    },

    // 3. 디바이스 전용 메모리 (접근 불가, 포인터만 유지)
    OpenClBuffer(usize),
    QnnTensor(usize),
}

// 포인터를 포함하므로 Thread Safety를 위해 마킹 (실제 구현 시 주의 필요)
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

#[derive(Debug, Clone)]
pub struct Tensor {
    name: Option<String>,
    shape: Shape,
    device: Device,
    // 데이터 소유권을 공유하기 위해 Arc 사용
    pub storage: Arc<Storage>,
}

impl Tensor {
    // [생성자] 일반 CPU 텐서
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        if data.len() != shape.num_elements() {
            panic!("Data size mismatch");
        }
        Self {
            name: None,
            shape,
            device: Device::Cpu,
            storage: Arc::new(Storage::Cpu(data)),
        }
    }

    // [생성자] 공유 메모리 텐서 (안드로이드 최적화)
    // 실제로는 AHardwareBuffer_allocate 등을 호출해야 함
    pub fn new_shared(shape: Shape) -> Self {
        let size_bytes = shape.num_elements() * 4;

        // Mockup: 실제로는 mmap된 포인터를 사용해야 함
        // 여기서는 벡터를 만들고 포인터만 취한 뒤 잊어버리는(forget) 흉내만 냅니다.
        let mut vec = vec![0u8; size_bytes];
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec); // 메모리 해제 방지 (실제 구현에선 소멸자 처리 필요)

        Self {
            name: None,
            shape,
            device: Device::Cpu, // 기본적으로 CPU에서 접근 가능
            storage: Arc::new(Storage::Shared {
                ptr,
                handle: 0, // Mock FD
                size: size_bytes,
            }),
        }
    }

    // [Helper] 0으로 초기화된 텐서
    pub fn zeros(shape: Shape) -> Self {
        let count = shape.num_elements();
        Self::new(vec![0.0; count], shape)
    }

    pub fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }

    // [속성] Shape 반환
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    // [데이터 접근] 읽기 전용 슬라이스
    // 주의: CPU 접근 가능한 스토리지일 때만 동작
    pub fn data(&self) -> &[f32] {
        match &*self.storage {
            Storage::Cpu(vec) => vec,
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            _ => panic!("Tensor data is not accessible on CPU directly"),
        }
    }

    // [데이터 접근] 쓰기 전용 슬라이스
    // 주의: Reference Count가 1일 때만 수정 가능 (CoW는 여기선 생략)
    pub fn data_mut(&mut self) -> &mut [f32] {
        // Arc의 유일한 소유자인지 확인하고, 아니면 복사(Clone)해야 하지만
        // 성능을 위해 여기서는 Unsafe하게 포인터를 따거나,
        // Storage가 Cpu일 때 get_mut을 시도합니다.

        // Shared Memory의 경우 항상 접근 허용
        if let Storage::Shared { ptr, size, .. } = *self.storage {
            return unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, size / 4) };
        }

        // CPU Memory의 경우
        match Arc::get_mut(&mut self.storage) {
            Some(Storage::Cpu(vec)) => vec,
            _ => panic!("Cannot mutate shared or non-cpu storage"),
        }
    }

    // [Offloading] 디바이스 이동
    pub fn to_device(&self, target: Device) -> Self {
        if self.device == target {
            return self.clone();
        }

        match (&*self.storage, target) {
            // Shared Memory는 장치만 바꾸면 됨 (Zero-Copy)
            (Storage::Shared { .. }, _) => {
                let mut new_tensor = self.clone();
                new_tensor.device = target;
                new_tensor
            }
            // CPU -> GPU/NPU (복사 필요)
            (Storage::Cpu(_), _) => {
                // TODO: 실제 백엔드 버퍼 할당 및 업로드 로직
                // 현재는 Mockup으로 CPU 유지
                let mut new_t = self.clone();
                new_t.device = target;
                new_t
            }
            _ => unimplemented!("Transfer not implemented"),
        }
    }

    // -------------------------------------------------------------------------
    // Backend Dispatcher & Operations
    // -------------------------------------------------------------------------

    fn backend(&self) -> Box<dyn Backend> {
        match self.device {
            Device::Cpu => Box::new(CpuBackend),
            _ => unimplemented!("Backend for device not implemented"),
        }
    }

    // A @ B
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let backend = self.backend();
        if self.device != other.device {
            let other_dev = other.to_device(self.device);
            return backend.matmul(self, &other_dev);
        }
        backend.matmul(self, other)
    }

    // A @ B^T
    pub fn matmul_transposed(&self, other: &Tensor) -> Tensor {
        let backend = self.backend();
        if self.device != other.device {
            let other_dev = other.to_device(self.device);
            return backend.matmul_transposed(self, &other_dev);
        }
        backend.matmul_transposed(self, other)
    }

    // A @ Slice^T (Zero-Copy)
    pub fn matmul_slice(&self, other_data: &[f32], rows: usize, cols: usize) -> Tensor {
        // Slice는 현재 CPU 메모리에 있다고 가정하므로,
        // Device가 GPU라면 임시 버퍼 업로드가 필요할 수 있음.
        // 현재는 CPU Backend에 최적화됨.
        self.backend().matmul_slice(self, other_data, rows, cols)
    }

    // In-Place Add
    pub fn add_assign(&mut self, other: &Tensor) {
        // 백엔드에서 In-Place 수정
        self.backend().add_assign(self, other);
    }

    // Helper for non-assign version
    pub fn add(&self, other: &Tensor) -> Tensor {
        let mut res = self.clone();
        res.add_assign(other);
        res
    }

    // Fused SiLU + Mul
    pub fn silu_mul_inplace(&mut self, other: &Tensor) {
        self.backend().silu_mul(self, other);
    }

    // RMS Norm
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        self.backend().rms_norm(self, weight, eps)
    }

    // Softmax
    pub fn softmax(&self) -> Tensor {
        self.backend().softmax(self)
    }

    // Scaling
    pub fn scale(&self, value: f32) -> Tensor {
        self.backend().scale(self, value)
    }

    // RoPE In-Place
    pub fn apply_rope_inplace(&mut self, start_pos: usize) {
        self.backend().rope_inplace(self, start_pos);
    }

    // Transpose
    // (Loader 최적화로 인해 사용 빈도는 줄었으나, fallback용으로 유지)
    pub fn transpose(&self) -> Tensor {
        // Transpose는 구조적 변경이므로 보통 Copy가 일어남
        // CPU 백엔드 구현을 사용하거나 각 백엔드별 구현 호출
        // 여기서는 CpuBackend의 구현이 있다고 가정하고 호출하거나 직접 구현
        // 편의상 CpuBackend 로직을 복사해 둠 (Backend trait에 transpose가 없다면)
        // 하지만 Backend trait에 copy_from 등이 있으므로,
        // 여기서는 간단히 CPU 로직으로 처리 (대부분 로딩 타임에만 쓰임)

        let dims = self.shape.dims();
        let (m, n) = (dims[0], dims[1]);
        let mut new_data = vec![0.0; m * n];
        let data = self.data(); // CPU access needed

        // Naive transpose (loading time only)
        for i in 0..m {
            for j in 0..n {
                new_data[j * m + i] = data[i * n + j];
            }
        }
        Tensor::new(new_data, Shape::new(vec![n, m]))
    }
}
