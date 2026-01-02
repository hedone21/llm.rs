use super::shape::Shape;
use std::sync::Arc;
// backend 모듈과 backend_impl 모듈이 core에 있다고 가정합니다.
use crate::backend::cpu::CpuBackend;
#[cfg(feature = "opencl")]
use crate::backend::opencl::OpenClBackend;
use crate::backend::{Backend, Device};

#[cfg(feature = "opencl")]
use ocl::{Buffer, Queue};

#[derive(Debug)]
pub enum Storage {
    Cpu(Vec<f32>),
    CpuQ4 {
        data: Vec<u8>,
        scales: Vec<f32>,
    },
    Shared {
        ptr: *mut u8,
        size: usize,
        #[cfg(feature = "opencl")]
        cl_buffer: Option<Buffer<f32>>, // Raw Bytes 컨테이너
        #[allow(dead_code)]
        #[cfg(feature = "opencl")]
        queue: Option<Queue>, // Unmap을 위해 큐 보관
    },
    #[allow(dead_code)]
    SharedQ4 {
        // 단일 버퍼에 [Data | Scales] 순서로 패킹하여 저장
        ptr: *mut u8,
        size: usize,
        #[cfg(feature = "opencl")]
        cl_buffer: Option<Buffer<f32>>, // Raw Bytes 컨테이너
        #[cfg(feature = "opencl")]
        cl_scales: Option<Buffer<f32>>, // Scale 부분의 버퍼 (별도 핸들)
        #[cfg(feature = "opencl")]
        queue: Option<Queue>,

        // 메타데이터
        data_len: usize,  // Q4 데이터 바이트 길이
        scale_len: usize, // Scale 데이터 바이트 길이 (실제 크기는 * 4)
    },
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
        #[cfg(feature = "opencl")]
        {
            // OpenCL 백엔드를 통해 Shared Memory 할당
            let backend = crate::backend::opencl::OpenClBackend::new();
            let mut tensor = backend.allocate_shared(&shape);
            tensor.device = Device::Cpu; // 초기 장치는 CPU로 설정
            tensor.data_mut().copy_from_slice(&data); // 데이터 복사
            tensor
        }
        #[cfg(not(feature = "opencl"))]
        {
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
    }

    // [Helper] 0으로 초기화된 텐서
    pub fn zeros(shape: Shape) -> Self {
        #[cfg(feature = "opencl")]
        {
            let backend = crate::backend::opencl::OpenClBackend::new();
            let mut tensor = backend.allocate_shared(&shape);
            tensor.device = Device::Cpu;
            tensor.data_mut().fill(0.0);
            tensor
        }
        #[cfg(not(feature = "opencl"))]
        {
            let count = shape.num_elements();
            Self::new(vec![0.0; count], shape)
        }
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

        #[cfg(feature = "opencl")]
        {
            let backend = crate::backend::opencl::OpenClBackend::new();
            // SharedQ4 저장소 할당
            let new_tensor = backend.allocate_shared_q4(&self.shape, q_data.len(), scales.len());

            match &*new_tensor.storage {
                Storage::SharedQ4 { ptr, data_len, .. } => unsafe {
                    let base_ptr = *ptr;
                    // Q4 데이터 및 스케일 복사
                    std::ptr::copy_nonoverlapping(q_data.as_ptr(), base_ptr, q_data.len());
                    let scale_ptr = base_ptr.add(*data_len) as *mut f32;
                    std::ptr::copy_nonoverlapping(scales.as_ptr(), scale_ptr, scales.len());
                },
                _ => unreachable!(),
            }
            new_tensor
        }
        #[cfg(not(feature = "opencl"))]
        {
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

    pub fn data(&self) -> &[f32] {
        match &*self.storage {
            Storage::Cpu(vec) => vec,
            Storage::Shared { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, size / 4)
            },
            Storage::SharedQ4 { ptr, size, .. } => unsafe {
                std::slice::from_raw_parts(*ptr as *const f32, *size / 4)
            },
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
                Storage::SharedQ4 { ptr, size, .. } => unsafe {
                    std::slice::from_raw_parts_mut(*ptr as *mut f32, *size / 4)
                },
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
            #[cfg(feature = "opencl")]
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
            #[cfg(feature = "opencl")]
            (Storage::CpuQ4 { data, scales }, Device::OpenCl) => {
                use crate::backend::opencl::OpenClBackend;
                let backend = OpenClBackend::new();

                // Q4용 Shared Buffer 할당 (백엔드에 요청)
                let new_tensor = backend.allocate_shared_q4(&self.shape, data.len(), scales.len());

                match &*new_tensor.storage {
                    Storage::SharedQ4 { ptr, data_len, .. } => {
                        unsafe {
                            let base_ptr = *ptr;

                            // 1. Q4 Data Copy
                            std::ptr::copy_nonoverlapping(data.as_ptr(), base_ptr, data.len());

                            // 2. Scales Copy (Data 뒤에 이어서 붙임)
                            // scales는 f32이므로 바이트 단위로 포인터 이동 후 복사
                            let scale_ptr = base_ptr.add(*data_len) as *mut f32;
                            std::ptr::copy_nonoverlapping(scales.as_ptr(), scale_ptr, scales.len());
                        }
                    }
                    _ => unreachable!("Must be SharedQ4"),
                }
                new_tensor
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
            #[cfg(feature = "opencl")]
            Device::OpenCl => Box::new(OpenClBackend::new()),
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
        if self.device != other.device {
            let other_dev = other.to_device(self.device);
            self.backend().add_assign(self, &other_dev);
            return;
        }

        // 백엔드에서 In-Place 수정
        self.backend().add_assign(self, other);
    }

    // Helper for non-assign version
    #[allow(dead_code)]
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
}
