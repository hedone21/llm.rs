// Rotary Positional Embedding (Split Layout 적용)
__kernel void kernel_rope(
    __global float * x,
    const int n_dims,     // head_dim (전체 차원)
    const int mode_n,     // head_dim / 2 (연산 루프 횟수)
    const int n_heads,    // 헤드 개수
    const int start_pos,
    const float freq_base,
    const float freq_scale
) {
    const int i0 = get_global_id(0); // 0..head_dim/2
    const int i1 = get_global_id(1); // Head index
    const int i2 = get_global_id(2); // Sequence index

    if (i0 >= mode_n) return;

    const int head_dim = n_dims;
    const int pos_idx = start_pos + i2;
    
    // Split Layout 인덱싱: x[i]와 x[i + mid]를 짝지음
    // x는 [Seq, HeadDim] 혹은 [Seq, Heads, HeadDim] 레이아웃 가정
    const long head_offset = i2 * (n_heads * head_dim) + i1 * head_dim;
    const long idx_0 = head_offset + i0;
    const long idx_1 = head_offset + i0 + mode_n;

    float theta = (float)pos_idx * pow(freq_base, -((float)(i0 * 2) / n_dims));
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float x0 = x[idx_0];
    float x1 = x[idx_1];

    x[idx_0] = x0 * cos_theta - x1 * sin_theta;
    x[idx_1] = x0 * sin_theta + x1 * cos_theta;
}
