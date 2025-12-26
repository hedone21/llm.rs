use std::any::Any;

use crate::backend::Device;

// 데이터 타입
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Int4,
}

// 모든 텐서 저장소의 기본 형태
#[derive(Debug, Clone)]
pub struct Storage {
    data: Vec<f32>,
    data_q4: Vec<u8>, // int4 데이터용
    scale: Vec<f32>,  // int4 스케일용

    dtype: DType,

    is_shared: bool, // shared memory인지 여부 (백엔드별 구현 시 활용)
}

impl Storage {
    // 저장소의 데이터를 바이트 단위 혹은 포인터로 접근 (Low-level)
    pub fn as_ptr<T>(&self) -> *const T {
        match self.dtype {
            DType::F32 => self.data.as_ptr() as *const T,
            DType::Int4 => self.data_q4.as_ptr() as *const T,
        }
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        match self.dtype {
            DType::F32 => self.data.as_mut_ptr() as *mut T,
            DType::Int4 => self.data_q4.as_mut_ptr() as *mut T,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    // 데이터 타입 정보 (f32, int4 등)
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // 저장소가 공유 메모리인지 여부
    pub fn is_shared(&self) -> bool {
        self.is_shared
    }

    // 다운캐스팅 지원 (백엔드 내부에서 구체적 핸들을 얻기 위함)
    pub fn as_any(&self) -> &dyn Any {
        self
    }

    // 저장소의 데이터를 OpenCl 버퍼로 반환 (GPU 메모리인 경우)
    #[cfg(feature = "opencl")]
    pub fn as_cl_buffer(&self) -> ocl::Buffer<f32> {
        unimplemented!();
    }

    #[cfg(feature = "opencl")]
    pub fn as_cl_buffer_mut(&mut self) -> ocl::Buffer<f32> {
        unimplemented!();
    }

    #[cfg(feature = "opencl")]
    pub fn queue(&self) -> &ocl::Queue {
        unimplemented!();
    }

    // CPU <-> GPU 간 캐시 동기화 (ARM 환경 필수)
    // CPU가 쓴 내용을 GPU가 보기 전, 혹은 그 반대에 호출
    #[cfg(feature = "opencl")]
    fn sync_to_device(&self) {
        unimplemented!();
    }
    #[cfg(feature = "opencl")]
    fn sync_to_host(&self) {
        unimplemented!();
    }
}

// 포인터를 포함하므로 Thread Safety를 위해 마킹 (실제 구현 시 주의 필요)
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

pub struct StorageBuilder {
    dtype: DType,

    capacity: usize,

    data: Option<Vec<f32>>,
    date_q4: Option<Vec<u8>>,
    scale: Option<Vec<f32>>,
    block_size: Option<usize>,

    is_shared: bool,
}

impl StorageBuilder {
    pub fn new(dtype: DType) -> Self {
        Self {
            dtype,
            capacity: 0,
            data: None,
            date_q4: None,
            scale: None,
            block_size: None,
            is_shared: false,
        }
    }

    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    pub fn data(mut self, data: Vec<f32>) -> Self {
        self.data = Some(data);
        self
    }

    pub fn data_q4(mut self, data_q4: Vec<u8>) -> Self {
        self.date_q4 = Some(data_q4);
        self
    }

    pub fn scale(mut self, scale: Vec<f32>, block_size: usize) -> Self {
        self.scale = Some(scale);
        self.block_size = Some(block_size);
        self
    }

    pub fn shared(mut self, is_shared: bool) -> Self {
        unimplemented!();
        self.is_shared = is_shared;
        self
    }

    pub fn build(self) -> Storage {
        match self.dtype {
            DType::F32 => Storage {
                data: self.data.unwrap_or_else(|| vec![0.0; self.capacity]),
                data_q4: vec![],
                scale: vec![],
                dtype: DType::F32,
                is_shared: self.is_shared,
            },
            DType::Int4 => Storage {
                data: vec![],
                data_q4: self
                    .date_q4
                    .unwrap_or_else(|| vec![0u8; (self.capacity + 1) / 2]),
                scale: self.scale.unwrap_or_else(|| {
                    vec![
                        1.0;
                        (self.capacity + self.block_size.unwrap_or(32) - 1)
                            / self.block_size.unwrap_or(32)
                    ]
                }),
                dtype: DType::Int4,
                is_shared: self.is_shared,
            },
        }
    }
}
