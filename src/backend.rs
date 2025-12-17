use crate::core::tensor::Tensor;

pub mod cpu;
pub mod opencl;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    Cpu,
    OpenCl, // GPU Offloading
    Qnn,    // NPU Offloading
}

// 모든 백엔드가 구현해야 할 연산 목록
pub trait Backend {
    fn device(&self) -> Device;
    fn name(&self) -> &str;

    // [기본 연산]
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor;

    // Linear 레이어 등에서 가중치 전치 없이 바로 연산하기 위함
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor) -> Tensor;

    // KV Cache 어텐션 연산 시 텐서 생성 없이 원본 데이터 슬라이스와 직접 연산 (Zero-Copy)
    // other_data: CPU 메모리 슬라이스 (Shared Memory일 경우 매핑된 포인터)
    fn matmul_slice(&self, a: &Tensor, other_data: &[f32], rows: usize, cols: usize) -> Tensor;

    // [In-Place 연산들]
    fn add_assign(&self, a: &mut Tensor, b: &Tensor);
    fn silu_mul(&self, gate: &mut Tensor, up: &Tensor);
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize);

    // [기타 연산]
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor;
    fn softmax(&self, x: &Tensor) -> Tensor;
    fn scale(&self, x: &Tensor, value: f32) -> Tensor;
    fn copy_from(&self, tensor: &Tensor) -> Tensor;
}
