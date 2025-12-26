__kernel void kernel_matmul_f32(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int M,
    int K,
    int N)
{
    // Simple Naive Implementation for Fallback/F32 support
    // A: [M, K], B: [N, K] (Transposed), C: [M, N]
    int r = get_global_id(0); // Row index of C (0..M)
    int c = get_global_id(1); // Col index of C (0..N)

    if (r < M && c < N) {
        float sum = 0.0f;
        for (int k=0; k<K; k++) {
            sum += A[r*K + k] * B[c*K + k];
        }
        C[r*N + c] = sum;
    }
}

__kernel void kernel_matmul_f32_transposed(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int M,
    int K,
    int N
) {
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r < M && c < N) {
        float sum = 0.0f;
        for (int k=0; k<K; k++) {
            sum += A[r*K + k] * B[c*K + k];
        }
        C[r*N + c] = sum;
    }
}
