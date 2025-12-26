__kernel void kernel_silu_mul(
    __global float * gate, 
    __global const float * up, 
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        float x = gate[i];
        float silu = x / (1.0f + exp(-x));
        gate[i] = silu * up[i];
    }
}

__kernel void kernel_add(__global float * dst, __global const float * src, const int n) {
    int i = get_global_id(0);
    if (i < n) dst[i] += src[i];
}

// Simple Softmax Fallback
__kernel void kernel_softmax_simple(
    __global const float * x, 
    __global float * dst, 
    const int ncols
) {
    int row = get_global_id(0);
    __global const float * x_row = x + row * ncols;
    __global float * dst_row = dst + row * ncols;
    
    float max_v = -1e30f;
    for(int i=0; i<ncols; i++) max_v = fmax(max_v, x_row[i]);
    
    float sum = 0.0f;
    for(int i=0; i<ncols; i++) {
        float val = exp(x_row[i] - max_v);
        dst_row[i] = val;
        sum += val;
    }
    float inv_sum = 1.0f / sum;
    for(int i=0; i<ncols; i++) dst_row[i] *= inv_sum;
}
