#include "layers/pooling_layers.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// MaxPooling kernel
__global__ void maxpool_kernel(
    const float* input, float* output,
    int B, int C, int in_h, int in_w,
    int out_h, int out_w,
    int pool_h, int pool_w, int stride)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z / C;
    int c = blockIdx.z % C;

    if (w_out < out_w && h_out < out_h) {
        float max_val = -1e30f;
        
        for (int ph = 0; ph < pool_h; ++ph) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int h_in = h_out * stride + ph;
                int w_in = w_out * stride + pw;
                
                if (h_in < in_h && w_in < in_w) {
                    int in_idx = ((b * C + c) * in_h + h_in) * in_w + w_in;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }
        
        int out_idx = ((b * C + c) * out_h + h_out) * out_w + w_out;
        output[out_idx] = max_val;
    }
}

void maxpool_forward_gpu(
    const float* d_input, float* d_output,
    int B, int C, int in_h, int in_w,
    int out_h, int out_w,
    int pool_h, int pool_w, int stride)
{
    dim3 block(16, 16, 1);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        B * C
    );
    
    maxpool_kernel<<<grid, block>>>(
        d_input, d_output,
        B, C, in_h, in_w, out_h, out_w,
        pool_h, pool_w, stride
    );
    
    cudaDeviceSynchronize();
}
