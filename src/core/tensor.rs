use super::shape::Shape;
use std::sync::Arc;
// backend 모듈과 backend_impl 모듈이 core에 있다고 가정합니다.
use crate::backend::cpu::CpuBackend;
use crate::backend::{Backend, Device};

use ocl::{Buffer, Queue};

#[derive(Debug)]
pub enum Storage {
    Cpu(Vec<f32>),
    CpuQ4 {
        data: Vec<u8>,
        scales: Vec<f32>,
    },
    // [수정] Shared Variant에 queue 필드 추가
    Shared {
        ptr: *mut u8,
        handle: usize,
        size: usize,
        cl_buffer: Option<Buffer<f32>>,
        queue: Option<Queue>, // Unmap을 위해 큐 보관
    },
    OpenCl(Buffer<f32>),
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

    // [Helper] 0으로 초기화된 텐서
    pub fn zeros(shape: Shape) -> Self {
        let count = shape.num_elements();
        Self::new(vec![0.0; count], shape)
    }

    pub fn quantize_q4(&self) -> Tensor {
        let data_f32 = self.data();

        let block_size = 32;
        let num_blocks = (data_f32.len() + block_size - 1) / block_size;

        // 데이터 크기는 절반으로 줄어듦 (32개 f32 -> 16개 u8)
        let mut q_data = Vec::with_capacity(data_f32.len() / 2);
        let mut scales = Vec::with_capacity(num_blocks);

        for chunk in data_f32.chunks(block_size) {
            let max_abs = chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            let scale = max_abs / 7.0;
            scales.push(scale);

            // [수정] 정확한 패킹 개수 계산 (홀수 길이 처리 포함)
            let num_pairs = (chunk.len() + 1) / 2;

            if scale == 0.0 {
                // 0x88 (0, 0) 값으로 채움
                q_data.extend(std::iter::repeat(0x88).take(num_pairs));
            } else {
                let inv_scale = 1.0 / scale;
                for pair in chunk.chunks(2) {
                    let v0 = pair[0];
                    let v1 = if pair.len() > 1 { pair[1] } else { 0.0 };

                    let q0 = ((v0 * inv_scale) + 8.5).clamp(0.0, 15.0) as u8;
                    let q1 = ((v1 * inv_scale) + 8.5).clamp(0.0, 15.0) as u8;
                    let packed = q0 | (q1 << 4);
                    q_data.push(packed);
                }
            }
        }

        Tensor {
            name: self.name.clone(),
            shape: self.shape.clone(),
            device: self.device,
            storage: Arc::new(Storage::CpuQ4 {
                data: q_data,
                scales,
            }),
        }
    }

    pub fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }

    // [속성] Shape 반환
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn from_storage(storage: Storage, shape: Shape, device: Device) -> Self {
        Self {
            name: None,
            shape,
            device,
            storage: Arc::new(storage),
        }
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn data(&self) -> &[f32] {
        match &*self.storage {
            Storage::Cpu(vec) => vec,
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            Storage::OpenCl(_) => {
                panic!("Cannot access Device-Local OpenCL tensor. Use .to_device(Cpu)")
            }
            _ => panic!("Tensor data is not accessible directly"),
        }
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        match Arc::get_mut(&mut self.storage) {
            Some(storage) => match storage {
                Storage::Cpu(vec) => vec,
                Storage::Shared { ptr, size, .. } => unsafe {
                    std::slice::from_raw_parts_mut(*ptr as *mut f32, *size / 4)
                },
                Storage::OpenCl(_) => {
                    panic!("Cannot access Device-Local OpenCL tensor. Use .to_device(Cpu)")
                }
                _ => panic!("Tensor data is not writable directly"),
            },
            None => {
                panic!("Cannot get mutable reference to Tensor data; multiple references exist")
            }
        }
    }

    pub fn to_device(&self, target: Device) -> Self {
        if self.device == target {
            return self.clone();
        }

        match (&*self.storage, target) {
            // 1. Shared -> Any (Zero-Copy)
            // 이미 Shared라면 디바이스 태그만 변경 (CPU <-> GPU 전환 비용 0)
            (Storage::Shared { .. }, _) => {
                let mut new_tensor = self.clone();
                new_tensor.device = target;
                new_tensor
            }

            // 2. Cpu -> OpenCl (Upload)
            (Storage::Cpu(_), Device::OpenCl) => {
                {
                    use crate::backend::opencl::OpenClBackend;
                    let backend = OpenClBackend::new();
                    let new_tensor = backend.allocate_shared(&self.shape);

                    // CPU -> Shared Mem Copy
                    let src = self.data();
                    let dst = unsafe {
                        match &*new_tensor.storage {
                            Storage::Shared { ptr, size, .. } => {
                                std::slice::from_raw_parts_mut(*ptr as *mut f32, *size / 4)
                            }
                            _ => unreachable!(),
                        }
                    };
                    dst.copy_from_slice(src);
                    new_tensor
                }
            }

            // 3. CpuQ4 -> OpenCl (Dequantize & Upload)
            // [중요] GPU는 현재 F32 커널만 지원하므로, 여기서 압축을 풉니다.
            (Storage::CpuQ4 { data, scales }, Device::OpenCl) => {
                {
                    use crate::backend::opencl::OpenClBackend;
                    let backend = OpenClBackend::new();
                    let new_tensor = backend.allocate_shared(&self.shape);

                    let dst = unsafe {
                        match &*new_tensor.storage {
                            Storage::Shared { ptr, size, .. } => {
                                std::slice::from_raw_parts_mut(*ptr as *mut f32, *size / 4)
                            }
                            _ => unreachable!(),
                        }
                    };

                    // Dequantize Loop (8.0 Offset Fix)
                    let block_size = 32;
                    let mut d_idx = 0;
                    for (i, &scale) in scales.iter().enumerate() {
                        let chunk_start = i * (block_size / 2);
                        let chunk_end = (chunk_start + (block_size / 2)).min(data.len());

                        for &packed in &data[chunk_start..chunk_end] {
                            let low = (packed & 0x0F) as f32;
                            let high = (packed >> 4) as f32;

                            // [Fix] 8.5 -> 8.0 (Correct Offset)
                            if d_idx < dst.len() {
                                dst[d_idx] = (low - 8.0) * scale;
                                d_idx += 1;
                            }
                            if d_idx < dst.len() {
                                dst[d_idx] = (high - 8.0) * scale;
                                d_idx += 1;
                            }
                        }
                    }
                    new_tensor
                }
            }

            // 4. Fallback (OpenCl -> Cpu Copy)
            (_, Device::Cpu) => {
                let data = self.data().to_vec();
                Tensor::new(data, self.shape.clone())
            }

            _ => unimplemented!("Transfer not implemented"),
        }
    }

    pub fn device(&self) -> Device {
        self.device
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
