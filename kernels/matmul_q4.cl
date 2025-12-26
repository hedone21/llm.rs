// Matrix-Vector Multiplication: Q4_0 (Matrix) x F32 (Vector)
// Optimized for Decoding (Batch Size = 1)
// ne00 = K (Hidden Size)
// ne01 = N (Output Size / Number of Rows)
__kernel void kernel_mul_mv_q4_0_f32(
    __global const uchar * src0_qs,     // Q4 indices (Matrix A)
    __global const float * src0_scales, // Scales (Matrix A)
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
    __global const float * scales_row = src0_scales + row * stride_scales;

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
