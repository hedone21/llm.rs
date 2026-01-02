// Matrix-Vector Multiplication: Q4_0 (Matrix) x F32 (Vector)
// Optimized for Decoding (Batch Size = 1)
// ne00 = K (Hidden Size)
// ne01 = N (Output Size / Number of Rows)
__kernel void kernel_mul_mv_q4_0_f32(
    __global const uchar * src0_qs,     // Q4 indices (Matrix A)
    __global const uchar * src0_scales, // Scales (Matrix A)
    __global const float * src1,        // Input vector (Vector B)
    __global float * dst,               // Output (Vector C)
    int ne00,                           
    int ne01                            
) {
    const int row = get_global_id(0);
    if (row >= ne01) return;

    const int nb = ne00 / QK4_0;
    
    // Planar Layout Strides
    // qs: ne00/2 bytes per row
    // scales: nb floats per row
    const int stride_qs = ne00 / 2; 
    const int stride_scales = nb;

    __global const uchar * qs_row = src0_qs + row * stride_qs;
    __global const float * scales_row = (__global const float *)src0_scales + row * stride_scales;

    float sum = 0.0f;

    // Loop over blocks (32 weights per block)
    for (int i = 0; i < nb; ++i) {
        float d = scales_row[i];
        int vec_idx = i * 32;

        // Process 16 bytes (32 nibbles)
        for (int j = 0; j < 16; ++j) {
            uchar packed = qs_row[i * 16 + j];
            float v0 = (float)(packed & 0x0F) - 8.0f;
            float v1 = (float)(packed >> 4) - 8.0f;

            sum += (v0 * d) * src1[vec_idx + 2*j];
            sum += (v1 * d) * src1[vec_idx + 2*j + 1];
        }
    }
    dst[row] = sum;
}

// Matrix-Matrix Multiplication: Matrix A (F32) x Matrix B (Q4_0, Transposed)
// A: [M, K], B: [N, K], C: [M, N]
__kernel void kernel_mul_mm_q4_0_f32(
    __global const uchar * src0_qs,     // Matrix B (Weights) [N, K/2]
    __global const uchar * src0_scales, // Matrix B (Scales) [N, K/32]
    __global const float * src1,        // Matrix A (Input) [M, K]
    __global float * dst,               // Matrix C (Output) [M, N]
    int ne00,                           // K (Hidden Size)
    int ne01,                           // N (Output Size)
    int ne11                            // M (Batch Size / Sequence Length)
) {
    const int col = get_global_id(0); // 0..N
    const int row = get_global_id(1); // 0..M

    if (col >= ne01 || row >= ne11) return;

    const int nb = ne00 / QK4_0;
    const int stride_qs = ne00 / 2;
    const int stride_scales = nb;

    __global const uchar * qs_row = src0_qs + col * stride_qs;
    __global const float * scales_row = (__global const float *)src0_scales + col * stride_scales;
    __global const float * src1_row = src1 + row * ne00;

    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        float d = scales_row[i];
        int vec_idx = i * 32;

        for (int j = 0; j < 16; ++j) {
            uchar packed = qs_row[i * 16 + j];
            float v0 = (float)(packed & 0x0F) - 8.0f;
            float v1 = (float)(packed >> 4) - 8.0f;
            sum += (v0 * d) * src1_row[vec_idx + 2*j];
            sum += (v1 * d) * src1_row[vec_idx + 2*j + 1];
        }
    }
    dst[row * ne01 + col] = sum;
}
