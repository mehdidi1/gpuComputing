#!/usr/bin/env python3
"""
PyTorch Model Timing Benchmark
Measures inference time on GPU and CPU for comparison with C++ implementation
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import sys

from export_pytorch_model import SimpleCNN


def load_test_data(input_file):
    """Load test input data from binary file"""
    data = np.fromfile(input_file, dtype=np.float32)
    # Determine number of samples based on file size
    # Each sample is 1x28x28 = 784 floats
    num_samples = len(data) // 784
    return data.reshape(num_samples, 1, 28, 28)


def benchmark_pytorch(model_path, input_path, device='cuda', batch_size=32, use_cudnn=True):
    """
    Benchmark PyTorch model inference time
    
    Args:
        model_path: Path to saved model
        input_path: Path to test input data
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        use_cudnn: If False, disables cuDNN optimizations (fair comparison)
    """
    print(f"PyTorch Benchmark on {device.upper()}")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Batch size: {batch_size}")
    print(f"cuDNN enabled: {use_cudnn}")
    print()
    
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Disable cuDNN for fair comparison with naive CUDA implementation
    if not use_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled - using standard PyTorch convolution")
    else:
        torch.backends.cudnn.enabled = True
    
    # Load model
    model = SimpleCNN()
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    
    model = model.to(device)
    model.eval()
    
    # Load test data
    input_data = load_test_data(input_path)
    num_samples = input_data.shape[0]
    print(f"Loaded {num_samples} samples")
    
    # Warm-up run
    with torch.no_grad():
        dummy_batch = torch.from_numpy(input_data[:batch_size]).to(device)
        _ = model(dummy_batch)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    total_time = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            # Prepare batch
            batch = torch.from_numpy(input_data[batch_start:batch_end]).to(device)
            
            # Time inference
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(batch)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            total_time += batch_time
            num_batches += 1
    
    print(f"\nTotal time: {total_time:.2f} ms")
    print(f"Average per batch ({batch_size} samples): {total_time / num_batches:.2f} ms")
    print(f"Average per sample: {total_time / num_samples:.2f} ms")
    print(f"Throughput: {num_samples / (total_time / 1000):.2f} samples/sec")
    
    return total_time


def main():
    # Default paths - try both python/ dir and workspace root
    script_dir = Path(__file__).parent
    
    model_path = script_dir / "exported_model" / "model_trained.pth"
    input_path = script_dir / "exported_model" / "test_data" / "test_input.bin"
    
    # Convert to strings
    model_path = str(model_path)
    input_path = str(input_path)
    
    print("="*70)
    print("PyTorch CNN Benchmark")
    print("="*70)
    print()
    
    # Benchmark on GPU with cuDNN (optimized)
    print(">>> WITH cuDNN Optimizations (PyTorch default) <<<")
    print()
    gpu_time_cudnn = benchmark_pytorch(model_path, input_path, device='cuda', batch_size=32, use_cudnn=True)
    
    print()
    print("="*70)
    print()
    
    # Benchmark on GPU WITHOUT cuDNN (fair comparison)
    print(">>> WITHOUT cuDNN (Fair comparison with naive CUDA) <<<")
    print()
    gpu_time_no_cudnn = benchmark_pytorch(model_path, input_path, device='cuda', batch_size=32, use_cudnn=False)
    
    print()
    print("="*70)
    print()
    
    # Benchmark on CPU
    print(">>> CPU Baseline <<<")
    print()
    cpu_time = benchmark_pytorch(model_path, input_path, device='cpu', batch_size=32, use_cudnn=False)
    
    print()
    print("="*70)
    print("Summary")
    print("="*70)
    print(f"GPU time (with cuDNN):    {gpu_time_cudnn:.2f} ms")
    print(f"GPU time (without cuDNN): {gpu_time_no_cudnn:.2f} ms")
    print(f"CPU time:                 {cpu_time:.2f} ms")
    print()
    print(f"Speedup with cuDNN (CPU/GPU):       {cpu_time/gpu_time_cudnn:.2f}x")
    print(f"Speedup without cuDNN (CPU/GPU):    {cpu_time/gpu_time_no_cudnn:.2f}x")
    print(f"cuDNN optimization factor (no-cuDNN/cuDNN): {gpu_time_no_cudnn/gpu_time_cudnn:.2f}x")
    print()
    print("For fair comparison with your C++ implementation, use the 'without cuDNN' time.")
    print()


if __name__ == "__main__":
    main()
