# ngp-baseline-torch

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **minimal, decoupled, high-performance PyTorch baseline** implementation of Instant Neural Graphics Primitives (Instant-NGP) for Neural Radiance Fields (NeRF).

This implementation prioritizes **clarity, modularity, and performance** with modern Python 3.13 and PyTorch 2.5+ features, providing a clean reference for research and production use.

## ✨ Features

- 🚀 **High Performance**: Optimized for modern CUDA GPUs (TF32, fused operations, efficient memory layout)
- 🧩 **Modular Design**: Strictly decoupled SISO (Simple-In-Simple-Out) modules
- 🎯 **Minimal Dependencies**: Only PyTorch and NumPy required
- 🔬 **Research Ready**: Comprehensive test suite with numerical validation
- 📦 **Production Ready**: Model export/import for deployment
- 🆕 **Modern Python**: Leverages Python 3.13+ and latest PyTorch features
- ⚡ **Mixed Precision**: Optional AMP support for faster training
- 🧪 **Well Tested**: 47 unit tests covering all components

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐
│   Ray Generator │  ← Camera poses & intrinsics
└────────┬────────┘
         │ RayBatch (SoA)
         ▼
┌─────────────────┐
│    Encoder      │  ← Positional Encoding or Hash Grid
└────────┬────────┘
         │ EncodedFeat
         ▼
┌─────────────────┐
│   Field (MLP)   │  ← Density & RGB features
└────────┬────────┘
         │ σ, RGB features
         ▼
┌─────────────────┐
│   RGB Head      │  ← View-dependent appearance
└────────┬────────┘
         │ RGB
         ▼
┌─────────────────┐
│  Integrator     │  ← Volume rendering
└────────┬────────┘
         │ Rendered RGB
         ▼
┌─────────────────┐
│   Loss & Opt    │  ← Training loop
└─────────────────┘
```

## 📋 Requirements

- Python ≥ 3.13
- PyTorch ≥ 2.5.0 (with CUDA 13.0)
- NumPy ≥ 2.0.0
- (Optional) pytest ≥ 8.0.0 for testing

## 🚀 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ngp-baseline-torch.git
cd ngp-baseline-torch

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev]"
```

### Quick Install

```bash
pip install torch>=2.5.0 numpy>=2.0.0
pip install -e .
```

## 📚 Usage

### Quick Start

```python
import torch
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all, create_optimizer
from ngp_baseline_torch.rays import load_nerf_synthetic, make_rays
from ngp_baseline_torch.runtime import Trainer
from ngp_baseline_torch.rng import seed_everything

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_everything(1337)

# Configuration
cfg = Config()
cfg.model.hash_levels = 16
cfg.model.mlp_width = 64
cfg.train.batch_rays = 4096

# Create model
encoder, field, rgb_head, occ_grid = create_all(cfg, device)
optimizer = create_optimizer(encoder, field, rgb_head, cfg)

# Load data
cameras = load_nerf_synthetic("data/nerf-synthetic/lego", "transforms_train.json")
cameras.poses = cameras.poses.to(device)
rays = make_rays(800, 800, cameras, None, device)

# Training
trainer = Trainer(encoder, field, rgb_head, optimizer, cfg, device, occ_grid)
metrics = trainer.step(ray_batch, target_rgb)
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

### Training Example

Run the provided training script:

```bash
python example_train.py --scene lego --iters 10000
```

### Configuration

All settings are controlled via dataclasses in `config.py`:

```python
from ngp_baseline_torch.config import Config

cfg = Config()

# Dataset
cfg.dataset.scene = "lego"
cfg.dataset.aabb = (-1.5, -1.5, -1.5, 1.5, 1.5, 1.5)

# Model architecture
cfg.model.hash_levels = 16
cfg.model.hash_res0 = 16
cfg.model.hash_per_level_scale = 1.5
cfg.model.mlp_width = 64
cfg.model.mlp_depth = 2

# Training
cfg.train.lr = 1e-2
cfg.train.batch_rays = 4096
cfg.train.iters = 20000

# Precision
cfg.precision.use_amp = True  # Enable mixed precision
```

### Rendering

```python
from ngp_baseline_torch.runtime import render_image

# Render a full image
image = render_image(
    H=800, W=800,
    pose=cameras.poses[0],
    focal=cameras.focal,
    near=cameras.near,
    far=cameras.far,
    encoder=encoder,
    field=field,
    rgb_head=rgb_head,
    cfg=cfg,
    device=device,
    chunk_size=4096
)

# Save image
import matplotlib.pyplot as plt
plt.imsave('output.png', image.cpu().numpy())
```

### Model Export

```python
from ngp_baseline_torch.artifact import export, load

# Export trained model
export(encoder, field, rgb_head, cfg, "outputs/lego", occ_grid)

# Load model
encoder2, field2, rgb_head2, _ = create_all(cfg, device)
encoder2, field2, rgb_head2, _ = load("outputs/lego", encoder2, field2, rgb_head2, device)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all quick tests
pytest -m quick

# Run with coverage
pytest --cov=ngp_baseline_torch

# Run specific test
pytest tests/test_encoder_hashgrid.py -v

# Run performance benchmarks (requires CUDA)
pytest -m perf
```

## 📊 Performance

On a modern NVIDIA GPU (RTX 3090/4090):

- **Training**: ~1-2M rays/second
- **Inference**: ~2-3M rays/second
- **Memory**: ~2-4GB for typical scenes
- **Convergence**: 20K iterations to 30+ dB PSNR

## 🗂️ Project Structure

```
ngp-baseline-torch/
├── src/ngp_baseline_torch/    # Main package
│   ├── types.py               # Type definitions & SISO contracts
│   ├── config.py              # Configuration dataclasses
│   ├── device.py              # Device management
│   ├── rng.py                 # Random number generation
│   ├── factory.py             # Component factory
│   ├── rays/                  # Ray generation
│   │   ├── cameras.py         # Camera data loading
│   │   └── rays.py            # Ray generation functions
│   ├── encoder/               # Feature encoders
│   │   ├── pe.py              # Positional encoding
│   │   └── hashgrid_torch.py  # Hash grid encoder
│   ├── field/                 # Neural fields
│   │   ├── mlp.py             # MLP networks
│   │   └── heads.py           # Output heads
│   ├── integrator/            # Volume rendering
│   │   ├── compositor.py      # Alpha compositing
│   │   └── marcher.py         # Ray marching
│   ├── grid/                  # Occupancy grid
│   │   └── occupancy.py       # Grid implementation
│   ├── loss/                  # Loss functions
│   │   └── rgb.py             # RGB losses
│   ├── opt/                   # Optimizers
│   │   └── adam.py            # Adam wrapper
│   ├── runtime/               # Training & inference
│   │   ├── train.py           # Training loop
│   │   └── infer.py           # Inference
│   └── artifact/              # Model serialization
│       └── export_v0.py       # Export/import
├── tests/                     # Comprehensive test suite
├── data/                      # Dataset directory
│   └── nerf-synthetic/        # NeRF synthetic scenes
├── example_train.py           # Training example
├── pyproject.toml             # Build configuration
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## 🎓 Key Concepts

### Structure-of-Arrays (SoA) Ray Format

Rays are stored in SoA format for GPU efficiency:

```python
@dataclass
class RayBatch:
    orig_x: Tensor  # [N]
    orig_y: Tensor  # [N]
    orig_z: Tensor  # [N]
    dir_x: Tensor   # [N]
    dir_y: Tensor   # [N]
    dir_z: Tensor   # [N]
    tmin: Tensor    # [N]
    tmax: Tensor    # [N]
```

### Multi-Resolution Hash Encoding

Instant-NGP's core innovation - multi-resolution hash grids:

- 16 resolution levels (16 → 2048)
- Trilinear interpolation
- Geometric progression of resolutions
- Collision-resistant hash function

### Volume Rendering

Standard NeRF volume rendering with optimizations:

- Early ray termination
- Efficient alpha compositing
- White background for synthetic scenes

## 🔬 Design Principles

1. **SISO (Simple-In-Simple-Out)**: Each function has clear inputs/outputs
2. **Modular**: Components can be swapped independently
3. **Testable**: Every module has unit tests
4. **Minimal**: No unnecessary abstractions
5. **Modern**: Uses latest Python/PyTorch features
6. **Fast**: Optimized for performance

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{ngp_baseline_torch2025,
  title = {NGP Baseline PyTorch: A Minimal Instant-NGP Implementation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/ngp-baseline-torch}
}
```

Original Instant-NGP paper:

```bibtex
@article{mueller2022instant,
  title={Instant neural graphics primitives with a multiresolution hash encoding},
  author={M{\"u}ller, Thomas and Evans, Alex and Schied, Christoph and Keller, Alexander},
  journal={ACM transactions on graphics (TOG)},
  volume={41},
  number={4},
  pages={1--15},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests before committing
pytest -m quick

# Code should pass all tests
pytest
```

### Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Keep modules decoupled
- Maintain SISO principle

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Instant-NGP](https://github.com/NVlabs/instant-ngp) by NVIDIA for the original implementation
- [NeRF](https://www.matthewtancik.com/nerf) for the foundational work on neural radiance fields
- [NeRF Synthetic Dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) for test scenes

## 📮 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/ngp-baseline-torch/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/ngp-baseline-torch/discussions)

## 🗺️ Roadmap

- [ ] CUDA kernel implementation for hash encoding
- [ ] Support for real-world datasets (COLMAP)
- [ ] Mesh extraction
- [ ] Multi-GPU training
- [ ] Web viewer
- [ ] More encoder variants (spherical harmonics, etc.)

## ⚡ Quick Links

- [Design Document](PROJECT_DESIGN.md)
- [API Documentation](docs/api.md)
- [Training Guide](docs/training.md)
- [Performance Tips](docs/performance.md)

---

**Note**: This is a research/educational baseline implementation. For production use with maximum performance, consider the official [Instant-NGP](https://github.com/NVlabs/instant-ngp) with custom CUDA kernels.
