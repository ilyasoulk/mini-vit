# Mini-ViT: An Image is Worth 4x4 Words

This repository contains a PyTorch implementation of smaller versions of the Vision Transformer (ViT) model introduced in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). While the original paper focuses on base, large, and huge architectures, this implementation explores lightweight variants (tiny and small) suitable for smaller datasets and computational resources.

## 🎯 Project Goals

The main objective of this project is to:
1. Implement a clean, understandable Vision Transformer from scratch
2. Create lightweight variants of the original architecture
3. Train and evaluate these models on CIFAR-10 and MNIST
4. Provide a learning resource for understanding transformer architectures in computer vision

## 🏗️ Model Architecture

This implementation includes two compact variants of the original ViT:

### ViT-Tiny
```python
{
    'patch_size': 4,
    'hidden_dim': 192,
    'fc_dim': 768,
    'num_heads': 3,
    'num_blocks': 12,
    'num_classes': 10
}
```

### ViT-Small (Current Implementation)
```python
{
    'patch_size': 4,
    'hidden_dim': 384,
    'fc_dim': 1536,
    'num_heads': 6,
    'num_blocks': 7,
    'num_classes': 10
}
```

Compare to original ViT-Base:
```python
{
    'patch_size': 16,
    'hidden_dim': 768,
    'fc_dim': 3072,
    'num_heads': 12,
    'num_blocks': 12,
    'num_classes': 1000
}
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ilyasoulk/mini-vit.git
cd mini-vit

# Install requirements
pip install -r requirements.txt
```

## 🚀 Usage

Train the model:
```bash
python src/train.py
```

## 📊 Results

Performance on MNIST:
- Test Accuracy: 98%
- Number of Parameters: 11 million
- Training Time: 2 Epochs on M2 Mac

Performance on CIFAR-10:
- Test Accuracy: 88%
- Number of Parameters: 11 million
- Training Time: 50 Epochs on NVIDIA T4

## 💡 Key Features

1. **Clean Implementation**: Each component of the Vision Transformer is implemented with clear, documented code:
   - Patch Embedding
   - Multi-Head Self-Attention
   - MLP Block
   - Position Embeddings

2. **Modifications for Smaller Scale**:
   - Smaller patch size (4x4 instead of 16x16)
   - Reduced model dimensions
   - Fewer attention heads
   - Fewer transformer blocks

3. **Training Optimizations**:
   - AdamW optimizer
   - Learning rate scheduling
   - Data augmentation for CIFAR-10

## 📁 Project Structure

```
mini-vit/
├── src/
│   ├── model.py    # ViT model implementation
│   └── train.py    # Training script
├── requirements.txt
└── README.md
```

## 🔍 Model Details

The implementation includes several key components:

1. **Patch Embedding**:
   - Divides input images into 4x4 patches
   - Projects patches to embedding dimension
   - Adds learnable position embeddings

2. **Transformer Encoder**:
   - Multi-head self-attention mechanism
   - Layer normalization
   - MLP block with GELU activation
   - Residual connections

## 🛠️ Technical Details

- Framework: PyTorch
- Dataset: CIFAR-10
- Hardware: NVIDIA T4
- Training Time: 50 Epochs

## 🔗 References

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
2. [Original ViT Implementation](https://github.com/google-research/vision_transformer)

## ⭐️ Show your support

Give a ⭐️ if this project helped you understand Vision Transformers!
