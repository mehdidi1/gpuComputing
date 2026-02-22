// ============================================================================
// Tensor Implementation
// ============================================================================
//

#include "tensor.h"
#include <random>
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// ============================================================================
// CONSTRUCTORS & DESTRUCTORS
// ============================================================================

Tensor::Tensor(int batch, int channels, int height, int width)
    : dims_({batch, channels, height, width}),
      gpu_data_(nullptr)
{
    total_size_ = batch * channels * height * width;
    data_ = std::make_unique<float[]>(total_size_);
    std::fill(data_.get(), data_.get() + total_size_, 0.0f); // initialize to 0
}

Tensor::Tensor(int batch, int features)
    : dims_({batch, features}),
      gpu_data_(nullptr)
{
    total_size_ = batch * features;
    data_ = std::make_unique<float[]>(total_size_);
    std::fill(data_.get(), data_.get() + total_size_, 0.0f); // initialize to 0
}

Tensor::Tensor(int batch, int channels, int height, int width, const float* data)
    : Tensor(batch, channels, height, width)
{
    std::copy(data, data + total_size_, data_.get());
}

Tensor::~Tensor()
{
    free_gpu();
    // CPU cleanup is handled automatically by std::unique_ptr
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : dims_(std::move(other.dims_)),
      total_size_(other.total_size_),
      data_(std::move(other.data_)),
      gpu_data_(other.gpu_data_)
{
    other.gpu_data_ = nullptr;
    other.total_size_ = 0;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other) {
        // Free our GPU memory
        free_gpu();
        
        // Move data from other
        dims_ = std::move(other.dims_);
        total_size_ = other.total_size_;
        data_ = std::move(other.data_);
        gpu_data_ = other.gpu_data_;
        
        // Clear other's GPU pointer
        other.gpu_data_ = nullptr;
        other.total_size_ = 0;
    }
    return *this;
}

Tensor Tensor::clone() const
{
    // Create new tensor with same dimensions
    Tensor copy = (dims_.size() == 4)
        ? Tensor(dims_[0], dims_[1], dims_[2], dims_[3])
        : Tensor(dims_[0], dims_[1]);

    // Copy all data
    std::copy(data_.get(), data_.get() + total_size_, copy.data_.get());

    return copy;
}

// ============================================================================
// ELEMENT ACCESS (4D)
// ============================================================================

float& Tensor::operator()(int b, int c, int h, int w)
{
    // Row-major order: index = b * (C*H*W) + c * (H*W) + h * W + w
    int index = b * dims_[1] * dims_[2] * dims_[3] + c * dims_[2] * dims_[3] + h * dims_[3] + w;
    return data_[index];
}

const float& Tensor::operator()(int b, int c, int h, int w) const
{
    int index = b * dims_[1] * dims_[2] * dims_[3] + c * dims_[2] * dims_[3] + h * dims_[3] + w;
    return data_[index];
}

// ============================================================================
// ELEMENT ACCESS (2D)
// ============================================================================

float& Tensor::operator()(int b, int f)
{
    // index = b * F + f
    int index = b * dims_[1] + f;
    return data_[index];
}

const float& Tensor::operator()(int b, int f) const
{
    int index = b * dims_[1] + f;
    return data_[index];
}

// ============================================================================
// UTILITY OPERATIONS
// ============================================================================

void Tensor::fill(float value)
{
    std::fill(data_.get(), data_.get() + total_size_, value);
}

void Tensor::zeros()
{
    fill(0.0f);
}

// ============================================================================
// GPU OPERATIONS
// ============================================================================

#include <cuda_runtime.h>

void Tensor::to_gpu()
{
    if (gpu_data_) return; // Already on GPU
    
    size_t bytes = total_size_ * sizeof(float);
    cudaMalloc(&gpu_data_, bytes);
    cudaMemcpy(gpu_data_, data_.get(), bytes, cudaMemcpyHostToDevice);
}

void Tensor::to_cpu()
{
    if (!gpu_data_) return; // Not on GPU
    
    size_t bytes = total_size_ * sizeof(float);
    cudaMemcpy(data_.get(), gpu_data_, bytes, cudaMemcpyDeviceToHost);
    cudaFree(gpu_data_);
    gpu_data_ = nullptr;
}

void Tensor::allocate_gpu()
{
    if (gpu_data_) return; // Already allocated
    
    size_t bytes = total_size_ * sizeof(float);
    cudaMalloc(&gpu_data_, bytes);
}

void Tensor::sync_to_cpu()
{
    if (!gpu_data_) return; // Not on GPU
    
    size_t bytes = total_size_ * sizeof(float);
    cudaMemcpy(data_.get(), gpu_data_, bytes, cudaMemcpyDeviceToHost);
}

void Tensor::free_gpu()
{
    if (gpu_data_) {
        cudaFree(gpu_data_);
        gpu_data_ = nullptr;
    }
}
