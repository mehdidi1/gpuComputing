// ============================================================================
// GPU Fully Connected Layer Implementation
// ============================================================================
#include <cuda_runtime.h>
#include <cmath>




__global__ void fc_forward_kernel(
    const float* __restrict__ d_input,      // [batch, in_features]
    const float* __restrict__ d_weights,    // [in_features, out_features]
    const float* __restrict__ d_bias,       // [out_features]
    float* __restrict__ d_output,           // [batch, out_features]
    int batch_size,
    int in_features,
    int out_features)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    extern __shared__ float smem_input[];
    
    int block_batch_start = blockIdx.x * blockDim.x;
    int num_rows = min(blockDim.x, batch_size - block_batch_start);
    
    // Load input rows into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int total_elements = num_rows * in_features;
    
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        int row = idx / in_features;
        int col = idx % in_features;
        int batch_idx = block_batch_start + row;
        
        if (batch_idx < batch_size) {
            smem_input[row * in_features + col] = d_input[batch_idx * in_features + col];
        }
    }
    
    __syncthreads();
    
    if (b < batch_size && j < out_features) {
        float sum = d_bias[j];
        
        int local_b = b - block_batch_start;
        
        // Dot product using cached input from shared memory
        for (int i = 0; i < in_features; ++i) {
            sum += smem_input[local_b * in_features + i] * d_weights[i * out_features + j];
        }
        
        d_output[b * out_features + j] = sum;
    }
}


void fc_forward_gpu(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float* d_output,
    int batch_size,
    int in_features,
    int out_features)
{
    dim3 block(16, 16);
    dim3 grid((batch_size + block.x - 1) / block.x,
              (out_features + block.y - 1) / block.y);
    
    // Shared memory for input: 16 rows × in_features floats
    size_t smem_bytes = 16 * in_features * sizeof(float);
    
    fc_forward_kernel<<<grid, block, smem_bytes>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_features, out_features);
}
