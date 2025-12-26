// Rotary Positional Embedding
__kernel void kernel_rope(
    __global float * x,
    const int n_dims,
    const int mode_n,
    const int n_heads, 
    const int start_pos,
    const float freq_base,
    const float freq_scale
) {
    const int i0 = get_global_id(0); // Index within head (0..head_dim/2)
    const int i1 = get_global_id(1); // Head index
    const int i2 = get_global_id(2); // Sequence index

    if (i0 >= mode_n) return;

    const int head_dim = mode_n * 2;
    const int pos_idx = start_pos + i2;
    const long offset = i2 * head_dim + i0 * 2; // Assuming contiguous within head for simplificty based on user tensor layout
    
    // Note: Rust tensor layout [Seq, Heads, HeadDim] or [Heads, Seq, HeadDim]
    // The previous implementation assumed simple contiguous memory access
    // This kernel assumes x points to the start of the head_dim chunk being processed.
    
    float theta = (float)pos_idx * pow(freq_base, -((float)(i0 * 2) / n_dims));
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    __global float * ptr = x + offset; 
    // WARNING: Offset logic depends heavily on Tensor Shape. 
    // In LlamaAttention, we extract heads first. So 'x' is usually [Seq, HeadDim].
    // If so, i1 (n_heads) is not used for offsetting if we launch per-head kernel.
    
    float x0 = ptr[0];
    float x1 = ptr[1];

    ptr[0] = x0 * cos_theta - x1 * sin_theta;
    ptr[1] = x0 * sin_theta + x1 * cos_theta;
}
