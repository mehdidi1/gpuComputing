#!/usr/bin/env python3
"""
============================================================================
PyTorch Model Validation - Compare C++ CNN output with PyTorch reference
============================================================================

This script:
1. Loads a trained PyTorch model
2. Runs inference on test data
3. Loads C++ inference results
4. Compares outputs and reports metrics

Usage:
    python3 validate_pytorch_model.py --cpp_output output.bin --pytorch_model model.pth


Requirements:
    pip install torch torchvision numpy
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import struct

from export_pytorch_model import SimpleCNN


# ============================================================================
# Output Validation Functions
# ============================================================================

def load_binary_output(filename, shape):
    """
    Load binary output file from C++ CNN
    
    - Open file
    - Read float32 values
    - Reshape to expected dimensions
    - Return as numpy array
    """
    print(f"Loading output from {filename}")
    data = np.fromfile(filename, dtype=np.float32)
    expected_size = int(np.prod(shape))
    if data.size != expected_size:
        raise ValueError(
            f"Size mismatch: expected {expected_size} floats but got {data.size}")
    return data.reshape(shape)


def load_pytorch_output(model, input_data, device="cpu"):
    """
    Generate reference output from PyTorch model
    
    """
    model.eval()
    model = model.to(device)
    input_tensor = torch.from_numpy(input_data).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return output.cpu().numpy()


def compute_metrics(cpp_output, pytorch_output):
    """
    Compare two outputs and compute error metrics
    
    Metrics to compute:
    - L2 error: sqrt(sum((cpp - pytorch)^2))
    - Max error: max(abs(cpp - pytorch))
    - Mean absolute error: mean(abs(cpp - pytorch))
    - Relative error: L2_error / L2_norm(pytorch)
    
    Returns:
        dict with all metrics
    """
    print("Computing error metrics...")

    diff = cpp_output - pytorch_output
    l2_error = float(np.sqrt(np.sum(diff ** 2)))
    max_error = float(np.max(np.abs(diff)))
    mean_abs_error = float(np.mean(np.abs(diff)))
    ref_norm = float(np.sqrt(np.sum(pytorch_output ** 2)))
    relative_error = float(l2_error / ref_norm) if ref_norm > 0 else float("inf")

    cpp_pred = np.argmax(cpp_output, axis=1)
    pytorch_pred = np.argmax(pytorch_output, axis=1)
    argmax_agreement = float(np.mean(cpp_pred == pytorch_pred))

    return {
        "l2_error": l2_error,
        "max_error": max_error,
        "mean_abs_error": mean_abs_error,
        "relative_error": relative_error,
        "argmax_agreement": argmax_agreement,
    }


def print_metrics(metrics, tolerance=1e-4):
    """
    Print validation metrics and overall status
    """
    print("\nValidation Results:")
    print("=" * 50)

    def status(ok):
        return "✓" if ok else "✗"

    max_ok = metrics["max_error"] <= tolerance
    mean_ok = metrics["mean_abs_error"] <= tolerance
    rel_ok = metrics["relative_error"] <= tolerance

    print(f"L2 Error         : {metrics['l2_error']:.6e}")
    print(f"Max Error        : {metrics['max_error']:.6e} {status(max_ok)}")
    print(f"Mean Abs Error   : {metrics['mean_abs_error']:.6e} {status(mean_ok)}")
    print(f"Relative Error   : {metrics['relative_error']:.6e} {status(rel_ok)}")
    print(f"Argmax Agreement : {metrics['argmax_agreement']:.2%}")

    overall = max_ok and mean_ok and rel_ok
    print("-" * 50)
    print(f"Overall Status   : {status(overall)}")
    print("=" * 50)


# ============================================================================
# Main Validation
# ============================================================================

def _parse_metadata(meta_path):
    meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, val = line.strip().split("=", 1)
            if key in {"input_shape", "output_shape"}:
                meta[key] = tuple(int(x) for x in val.strip()[1:-1].split(",") if x.strip())
            elif key == "num_samples":
                meta[key] = int(val)
    return meta


def _load_input_data(input_file):
    if input_file is None:
        return None

    path = Path(input_file)
    if path.suffix == ".npy":
        data = np.load(path)
        return data.astype(np.float32)

    if path.suffix == ".npz":
        with np.load(path) as npz:
            if "input" not in npz:
                raise ValueError(".npz must contain 'input' array")
            return npz["input"].astype(np.float32)

    if path.suffix == ".bin":
        meta_path = path.parent / "metadata.txt"
        if not meta_path.exists():
            raise FileNotFoundError("metadata.txt not found in input directory")
        meta = _parse_metadata(meta_path)
        input_shape = meta["input_shape"]
        data = np.fromfile(path, dtype=np.float32)
        expected_size = int(np.prod(input_shape))
        if data.size != expected_size:
            raise ValueError(
                f"Input size mismatch: expected {expected_size} floats but got {data.size}")
        return data.reshape(input_shape)

    raise ValueError("Unsupported input file format")

def validate_against_pytorch(cpp_output_file, pytorch_model_file, 
                            input_file=None, tolerance=1e-4):
    """
    Main validation function
    
    Steps:
    1. Load PyTorch model
    2. Load test input (or create dummy input)
    3. Generate PyTorch reference output
    4. Load C++ output
    5. Compare and report metrics
    """
    print("PyTorch Validation")
    print("=" * 50)

    # 1. Load PyTorch model
    print(f"\n1. Loading PyTorch model from {pytorch_model_file}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    state = torch.load(pytorch_model_file, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state

    # 2. Load test input or create dummy
    print(f"\n2. Loading test input...")
    if input_file:
        print(f"   From file: {input_file}")
        input_data = _load_input_data(input_file)
    else:
        print("   Creating dummy input [1, 1, 28, 28]")
        input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

    # 3. Generate PyTorch output
    print("\n3. Running PyTorch inference...")
    pytorch_output = load_pytorch_output(model, input_data, device=device)

    # 4. Load C++ output
    print(f"\n4. Loading C++ output from {cpp_output_file}...")
    cpp_output = load_binary_output(cpp_output_file, pytorch_output.shape)

    # 5. Compare
    print("\n5. Comparing outputs...")
    metrics = compute_metrics(cpp_output, pytorch_output)
    print_metrics(metrics, tolerance)


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate C++ CNN implementation against PyTorch"
    )
    parser.add_argument("--cpp_output", required=True,
                       help="Path to C++ inference output (binary)")
    parser.add_argument("--pytorch_model", required=True,
                       help="Path to PyTorch model (.pth)")
    parser.add_argument("--input", default=None,
                       help="Path to test input data (optional)")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                       help="Error tolerance threshold")
    
    args = parser.parse_args()
    
    try:
        validate_against_pytorch(args.cpp_output, args.pytorch_model,
                                args.input, args.tolerance)
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
