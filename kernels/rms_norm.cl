// RMS Norm: Block-based reduction
__kernel void kernel_rms_norm(
    __global const float * x,
    __global const float * w,
    __global float * dst,
    const int ncols,
    const float eps,
    __local float * sum_cache) 
{
    const int row = get_group_id(0);
    const int tid = get_local_id(0);
    const int block_size = get_local_size(0);

    __global const float * x_row = x + row * ncols;
    __global float * dst_row = dst + row * ncols;

    float tmp = 0.0f;
    for (int i = tid; i < ncols; i += block_size) {
        float xi = x_row[i];
        tmp += xi * xi;
    }
    
    // Local Reduction
    sum_cache[tid] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_cache[tid] += sum_cache[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean = sum_cache[0] / ncols;
    float scale = 1.0f / sqrt(mean + eps);

    for (int i = tid; i < ncols; i += block_size) {
        dst_row[i] = x_row[i] * scale * w[i];
    }
}
