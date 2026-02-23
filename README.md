# CNN MNIST Inference

## How to Run

```bash
cd build
./cnn_app [cpu|gpu] [num_samples]
```

**Examples:**
```bash
./cnn_app gpu 1000    # Process 1000 samples on GPU
./cnn_app cpu 100     # Process 100 samples on CPU
./cnn_app             # Default: GPU mode, 100 samples
```
