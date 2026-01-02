__kernel void kernel_copy(__global float* src, __global float* dst, int n) {
    int i = get_global_id(0);
    if (i < n) dst[i] = src[i];
}
