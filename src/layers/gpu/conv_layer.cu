#include "layers/conv_layer_gpu.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>


__global__ void conv2d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ kernels,
    const float* __restrict__ bias,  
    float* __restrict__ output,        
    int B, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride, int padding)
{
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Z dimension handles both Batch and Output Channels
    int b = blockIdx.z / out_c;
    int k = blockIdx.z % out_c;

    int block_h = blockDim.y;
    int block_w = blockDim.x;
    int h_base = blockIdx.y * block_h * stride - padding;
    int w_base = blockIdx.x * block_w * stride - padding;
    int tile_h = block_h * stride + kernel_h - 1;
    int tile_w = block_w * stride + kernel_w - 1;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int tile_size = tile_h * tile_w;

    // Shared memory for the input (if available)
    extern __shared__ float smem[];
    
    bool use_smem = (blockDim.x * blockDim.y <= 256) && (in_c * tile_h * tile_w <= 6144); 
    
    if (use_smem && w_out < out_w && h_out < out_h) {
        for (int c = 0; c < in_c; ++c) {
            float* smem_channel = smem + c * tile_size;
            
            // Distribute loading across all threads in the block
            for (int elem_idx = tid; elem_idx < tile_size; elem_idx += num_threads) {
                int h_tile = elem_idx / tile_w;
                int w_tile = elem_idx % tile_w;
                
                int h_in = h_base + h_tile;
                int w_in = w_base + w_tile;
                
                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    int in_idx = ((b * in_c + c) * in_h + h_in) * in_w + w_in;
                    smem_channel[elem_idx] = input[in_idx];
                } else {
                    smem_channel[elem_idx] = 0.0f;
                }
            }
            __syncthreads();
        }
        
        // Compute convolution using cached input
        float sum = bias[k];

        for (int c = 0; c < in_c; ++c) {
            float* smem_channel = smem + c * tile_size;
            
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_tile = threadIdx.y * stride + kh;
                    int w_tile = threadIdx.x * stride + kw;
                    
                    if (h_tile < tile_h && w_tile < tile_w) {
                        int tile_idx = h_tile * tile_w + w_tile;
                        int k_idx = ((k * in_c + c) * kernel_h + kh) * kernel_w + kw;
                        
                        sum += smem_channel[tile_idx] * kernels[k_idx];
                    }
                }
            }
        }
        
        // Single write to global memory
        int out_idx = ((b * out_c + k) * out_h + h_out) * out_w + w_out;
        output[out_idx] = sum;
    } else if (w_out < out_w && h_out < out_h) {
        // Fallback: use global memory (original kernel)
        float sum = bias[k];

        for (int c = 0; c < in_c; ++c) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int in_idx = ((b * in_c + c) * in_h + h_in) * in_w + w_in;
                        int k_idx = ((k * in_c + c) * kernel_h + kh) * kernel_w + kw;
                        
                        sum += input[in_idx] * kernels[k_idx];
                    }
                }
            }
        }
        
        int out_idx = ((b * out_c + k) * out_h + h_out) * out_w + w_out;
        output[out_idx] = sum;
    }
}

// ============================================================================
// ConvolutionLayerGPU Implementation
// ============================================================================

ConvolutionLayerGPU::ConvolutionLayerGPU(int num_filters, int kernel_h, int kernel_w,
                                         int stride, int padding, int in_channels)
    : num_filters_(num_filters),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_(stride),
      padding_(padding),
      in_channels_(in_channels),
      d_kernels_(nullptr),
      d_bias_(nullptr),
      kernels_(0, 0, 0, 0),
      bias_(num_filters, 0.0f)
{

}

ConvolutionLayerGPU::~ConvolutionLayerGPU()
{
    if (d_kernels_) cudaFree(d_kernels_);
    if (d_bias_) cudaFree(d_bias_);
}

void ConvolutionLayerGPU::forward(const Tensor& input, Tensor& output)
{
    const auto in_shape = input.get_shape();
    int B = in_shape[0];
    int in_h = in_shape[2];
    int in_w = in_shape[3];

    int out_h = (in_h + 2 * padding_ - kernel_h_) / stride_ + 1;
    int out_w = (in_w + 2 * padding_ - kernel_w_) / stride_ + 1;

    // Ensure input is on GPU
    if (!input.is_on_gpu()) {
        const_cast<Tensor&>(input).to_gpu();
    }
    
    // Ensure output has GPU memory allocated
    if (!output.is_on_gpu()) {
        output.allocate_gpu();
    }

    dim3 block(16, 16, 1); 
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        B * num_filters_
    );

    int tile_h = block.y * stride_ + kernel_h_ - 1;
    int tile_w = block.x * stride_ + kernel_w_ - 1;
    size_t smem_bytes = (size_t)in_channels_ * tile_h * tile_w * sizeof(float);
    
    // Check if shared memory fits
    if (smem_bytes > 96 * 1024) {
        smem_bytes = 0;
    }

    conv2d_kernel<<<grid, block, smem_bytes>>>(
        input.gpu_data(), d_kernels_, d_bias_, output.gpu_data(),
        B, in_channels_, in_h, in_w,
        num_filters_, out_h, out_w,
        kernel_h_, kernel_w_, stride_, padding_
    );

    cudaDeviceSynchronize();
    
}

std::vector<int> ConvolutionLayerGPU::get_output_shape(const std::vector<int>& input_shape) const
{
    int B = input_shape[0];
    int H = input_shape[2];
    int W = input_shape[3];
    
    int out_h = (H + 2 * padding_ - kernel_h_) / stride_ + 1;
    int out_w = (W + 2 * padding_ - kernel_w_) / stride_ + 1;
    
    return {B, num_filters_, out_h, out_w};
}

void ConvolutionLayerGPU::set_weights(const Tensor& kernels)
{
    kernels_ = kernels.clone(); // Store CPU backup
    
    size_t bytes = num_filters_ * in_channels_ * kernel_h_ * kernel_w_ * sizeof(float);
    
    if (d_kernels_) cudaFree(d_kernels_);
    cudaMalloc(&d_kernels_, bytes);
    cudaMemcpy(d_kernels_, kernels_.data(), bytes, cudaMemcpyHostToDevice);
}

void ConvolutionLayerGPU::set_bias(const std::vector<float>& bias)
{
    bias_ = bias; // Store CPU backup
    
    size_t bytes = num_filters_ * sizeof(float);
    
    if (d_bias_) cudaFree(d_bias_);
    cudaMalloc(&d_bias_, bytes);
    cudaMemcpy(d_bias_, bias_.data(), bytes, cudaMemcpyHostToDevice);
}