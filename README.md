# Flash-Enhanced Optimized U-Net: High-Performance Image Segmentation

A highly optimized PyTorch implementation of U-Net architecture featuring Flash Attention and modern deep learning optimizations. This implementation is specifically tuned for the NVIDIA RTX 4070 GPU and can be used for various image segmentation tasks.

## Key Features

- Flash Attention integration for faster attention computation
- Mixed precision training (FP16)
- Memory-efficient gradient checkpointing
- Optimized for RTX 4070's tensor cores
- Automatic dataset handling via Hugging Face
- Real-time visualization of training progress
- Depth-wise separable convolutions for efficiency

## Prerequisites

### Hardware Requirements
- GPU: Optimized for NVIDIA RTX 4070 (12GB VRAM)
- RAM: Minimum 16GB recommended
- Storage: Space for dataset and checkpoints

Note: If you're using a different GPU, you might need to adjust the batch size and other parameters accordingly. The current settings are optimized for RTX 4070's 12GB VRAM.

### Software Requirements
- Python 3.8 or higher
- CUDA 11.7 or higher

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn einops datasets tqdm matplotlib numpy
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Load the dataset from Hugging Face
2. Initialize the optimized U-Net model
3. Start training with real-time visualization
4. Display both training and validation metrics

## Optimizations Explained

### Memory Optimizations
- Selective Attention: Only used in deeper layers where it matters most
- Gradient Checkpointing: Applied selectively to manage memory usage
- Batch Size: Optimized to 8 for RTX 4070's 12GB VRAM
- Mixed Precision: Uses FP16 for efficient memory usage

### Performance Optimizations
- Flash Attention: Faster and more memory-efficient attention mechanism
- Depth-wise Separable Convolutions: More efficient than standard convolutions
- Tensor Core Utilization: Channel sizes aligned for optimal tensor core usage
- Efficient Data Loading: Optimized worker count and memory pinning

### Training Optimizations
- AdamW Optimizer: Modern optimizer with proper weight decay
- OneCycleLR: Advanced learning rate scheduling
- Efficient Gradient Clearing: Uses memory-efficient gradient clearing
- Compiled Model: Uses torch.compile with "max-autotune" mode

## Configuration

Key parameters you might want to adjust based on your setup:
```python
BATCH_SIZE = 8  # Adjust based on your GPU memory
NUM_WORKERS = 4  # Adjust based on your CPU cores
MIXED_PRECISION = True  # Set to False if you encounter issues
```

## Dataset

The project uses Hugging Face datasets. To use your own dataset:
1. Upload it to Hugging Face in the correct format
2. Modify the dataset path in the code
3. Ensure your data follows the expected format (image and label pairs)

## Common Issues & Solutions

1. Out of Memory (OOM)
   - Reduce batch size
   - Disable some attention layers
   - Reduce model size

2. Slow Training
   - Check CUDA version compatibility
   - Adjust number of workers
   - Ensure proper tensor core utilization

3. Poor Convergence
   - Adjust learning rate
   - Modify OneCycleLR parameters
   - Check data normalization

## Troubleshooting

If you encounter CUDA out of memory errors:
```python
# Try reducing batch size
BATCH_SIZE = 4  # Instead of 8

# Or disable attention in more layers
self.conv_down_2 = OptimizedResBlock(8, 16, use_attention=False)
```

## Contributing

Feel free to:
- Report issues
- Suggest optimizations
- Submit pull requests
- Share your results

## License

This project is MIT licensed. Feel free to use it in your own projects.

## Acknowledgments

- Flash Attention implementation based on tri-dao's work
- U-Net architecture inspired by the original paper
- Optimizations tuned for modern GPU architectures
