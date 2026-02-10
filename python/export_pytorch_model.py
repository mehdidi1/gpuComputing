#!/usr/bin/env python3
"""
============================================================================
PyTorch Model Exporter - Convert PyTorch models to CNN framework format
============================================================================

This script:
1. Defines a PyTorch CNN model
2. Trains it (or loads pre-trained weights)
3. Exports weights to a binary format readable by C++
4. Saves test inputs/outputs for validation

TODO: Implement weight exporting and data validation

Requirements:
    pip install torch torchvision numpy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
import struct
import sys


# ============================================================================
# PyTorch CNN Model Definition
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for MNIST classification
    
    Architecture:
        Input: [batch, 1, 28, 28]
        Conv(32, 5x5, stride=1, padding=2) -> ReLU -> MaxPool(2x2)
        Conv(64, 5x5, stride=1, padding=2) -> ReLU -> MaxPool(2x2)
        Flatten
        FC(128) -> ReLU
        FC(10) -> Softmax
        Output: [batch, 10]
    
    TODO: Modify this architecture as needed for your experiments
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After two 2x2 pooling layers, 28x28 becomes 7x7
        # 64 channels * 7 * 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# Binary Weight Export Functions
# ============================================================================

class WeightExporter:
    """
    Exports PyTorch model weights to custom binary format
    
    Binary format structure:
    [HEADER]
    - Magic number: "CNN!" (4 bytes)
    - Version: uint32 = 1
    - Num layers: uint32
    
    [LAYER 0]
    - Layer type: uint32 (0=Conv, 1=FC, etc.)
    - Layer name: string (null-terminated)
    - Weight shape: [num_dims] uint32 values
    - Weight data: float32 array
    - Bias shape: uint32
    - Bias data: float32 array
    
    [LAYER 1]
    ... (repeat for each layer)
    
    TODO: Implement the export functions
    """
    
    @staticmethod
    def write_string(f, s):
        """Write null-terminated string to file"""
        data = s.encode("utf-8") + b"\x00"
        f.write(data)
    
    @staticmethod
    def write_floats(f, arr):
        """Write float array to file (binary format)"""
        arr_f32 = np.asarray(arr, dtype=np.float32)
        f.write(arr_f32.tobytes(order="C"))
    
    @staticmethod
    def export_model(model, output_path):
        """
        Export PyTorch model to binary format
        
        TODO: Implement full export
        Steps:
        1. Open file for binary writing
        2. Write header (magic, version, num_layers)
        3. For each layer in model:
           a. Get layer type (Conv2d, Linear, etc.)
           b. Extract weights and biases
           c. Write layer data to file
        4. Close file
        """
        print(f"Exporting model to {output_path}")

        layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layers.append((name, module))

        with open(output_path, "wb") as f:
            # Header
            f.write(b"CNN!")
            f.write(struct.pack("<I", 1))  # version
            f.write(struct.pack("<I", len(layers)))

            for name, module in layers:
                if isinstance(module, nn.Conv2d):
                    layer_type = 0
                    weights = module.weight.detach().cpu().numpy().astype(np.float32)
                    bias = (
                        module.bias.detach().cpu().numpy().astype(np.float32)
                        if module.bias is not None
                        else np.zeros((weights.shape[0],), dtype=np.float32)
                    )

                    out_c, in_c, kh, kw = weights.shape
                    f.write(struct.pack("<I", layer_type))
                    f.write(struct.pack("<4I", out_c, in_c, kh, kw))
                    WeightExporter.write_floats(f, weights)

                    f.write(struct.pack("<I", bias.shape[0]))
                    WeightExporter.write_floats(f, bias)

                elif isinstance(module, nn.Linear):
                    layer_type = 1
                    weight = module.weight.detach().cpu().numpy().astype(np.float32)
                    bias = (
                        module.bias.detach().cpu().numpy().astype(np.float32)
                        if module.bias is not None
                        else np.zeros((weight.shape[0],), dtype=np.float32)
                    )

                    # PyTorch Linear weights are [out_features, in_features]
                    # C++ expects [in_features, out_features]
                    weight_t = weight.T
                    in_f, out_f = weight_t.shape

                    f.write(struct.pack("<I", layer_type))
                    f.write(struct.pack("<4I", in_f, out_f, 1, 1))
                    WeightExporter.write_floats(f, weight_t)

                    f.write(struct.pack("<I", bias.shape[0]))
                    WeightExporter.write_floats(f, bias)

        print(f"Exported {len(layers)} weight layers")


# ============================================================================
# Data Loading & Training
# ============================================================================

def load_mnist_data(batch_size=32, data_dir="./data"):
    """
    Load MNIST dataset and return DataLoaders
    
    TODO: Implement MNIST loading
    Returns:
        train_loader, test_loader - PyTorch DataLoaders
    """
    print("Loading MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model, train_loader, num_epochs=5):
    """
    Train the PyTorch model
    
    TODO: Implement training loop
    - Use CrossEntropyLoss and Adam optimizer
    - Print loss after each epoch
    - Save model checkpoint
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")


def evaluate_model(model, test_loader):
    """
    Evaluate model on test set
    
    TODO: Implement evaluation
    - Compute accuracy and loss
    - Print metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.numel()

    avg_loss = total_loss / max(1, len(test_loader))
    accuracy = correct / max(1, total)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")


# ============================================================================
# Test Data Export for Validation
# ============================================================================

def export_test_data(model, test_loader, output_dir="./test_data"):
    """
    Export test inputs and outputs for validation against C++ implementation
    
    TODO: Implement test data export
    
    Export format:
    - test_input.bin: Input tensors [num_samples, 1, 28, 28] as float32
    - test_output.bin: Model outputs [num_samples, 10] as float32
    - metadata.txt: num_samples, input_shape, output_shape
    
    These files allow C++ code to compare outputs with PyTorch reference
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting test data to {output_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    inputs = []
    outputs = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            out = model(data)

            inputs.append(data.cpu().numpy().astype(np.float32))
            outputs.append(out.cpu().numpy().astype(np.float32))

    input_arr = np.concatenate(inputs, axis=0)
    output_arr = np.concatenate(outputs, axis=0)

    input_path = Path(output_dir) / "test_input.bin"
    output_path = Path(output_dir) / "test_output.bin"
    meta_path = Path(output_dir) / "metadata.txt"

    input_arr.tofile(input_path)
    output_arr.tofile(output_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"num_samples={input_arr.shape[0]}\n")
        f.write(f"input_shape={input_arr.shape}\n")
        f.write(f"output_shape={output_arr.shape}\n")

    print(f"Saved inputs to {input_path}")
    print(f"Saved outputs to {output_path}")
    print(f"Saved metadata to {meta_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main execution flow:
    1. Create model
    2. Load or train weights
    3. Export to binary format
    4. Export test data for validation
    """
    print("PyTorch CNN Framework - Model Exporter")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./exported_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. Create model
    # ========================================================================
    print("\n1. Creating SimpleCNN model...")
    model = SimpleCNN()
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ========================================================================
    # 2. Load data
    # ========================================================================
    print("\n2. Loading MNIST data...")
    train_loader, test_loader = load_mnist_data(batch_size=32)
    
    # ========================================================================
    # 3. Train or load model
    # ========================================================================
    print("\n3. Training model (this will take a while)...")
    train_model(model, train_loader, num_epochs=3)
    torch.save(model.state_dict(), output_dir / "model_trained.pth")
    print("   ✓ Model trained and saved")
    
    # ========================================================================
    # 4. Evaluate model
    # ========================================================================
    print("\n4. Evaluating model...")
    evaluate_model(model, test_loader)
    
    # ========================================================================
    # 5. Export model weights
    # ========================================================================
    print("\n5. Exporting model weights...")
    try:
        exporter = WeightExporter()
        exporter.export_model(model, output_dir / "model_weights.bin")
        print("   ✓ Model weights exported")
    except NotImplementedError as e:
        print(f"   ✗ {e}")
    
    # ========================================================================
    # 6. Export test data for validation
    # ========================================================================
    print("\n6. Exporting test data for validation...")
    try:
        export_test_data(model, test_loader, output_dir / "test_data")
        print("   ✓ Test data exported")
    except Exception as e:
        print(f"   ✗ {e}")
    
    print("\n" + "=" * 50)
    print("✓ Model training and export pipeline complete!")
    print(f"   Weights saved to: {output_dir}/model_weights.bin")
    print(f"   Checkpoint saved to: {output_dir}/model_trained.pth")
    print(f"   Test data saved to: {output_dir}/test_data/")
    print("=" * 50)


if __name__ == "__main__":
    main()
