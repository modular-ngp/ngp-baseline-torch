"""Configuration dataclasses for NGP baseline."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset configuration for NeRF synthetic."""
    root: str = "data/nerf-synthetic"
    scene: str = "lego"
    train_split: str = "transforms_train.json"
    val_split: str = "transforms_val.json"
    test_split: str = "transforms_test.json"
    scale: float = 1.0
    aabb: tuple[float, float, float, float, float, float] = (-1.5, -1.5, -1.5, 1.5, 1.5, 1.5)
    white_background: bool = True  # Use white background for NeRF-Synthetic


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    pe_bands: int = 10
    hash_levels: int = 16
    hash_res0: int = 16
    hash_per_level_scale: float = 1.5
    hash_features_per_level: int = 2
    mlp_width: int = 64
    mlp_depth: int = 2
    activation: str = "relu"
    view_dependent: bool = True  # CRITICAL: Enable view-dependent colors
    pe_viewdir_bands: int = 4
    density_activation: str = "trunc_exp"  # Use truncated exponential for density


@dataclass
class IntegratorConfig:
    """Volume integration configuration."""
    step_strategy: str = "fixed"  # "fixed" or "grid"
    n_steps_fixed: int = 128  # More samples for better quality
    sigma_thresh: float = 0.01  # Enable early stopping
    early_stop_T: float = 1e-4  # Enable early stopping
    dt_gamma: float = 0.0  # for adaptive step size
    perturb: bool = True  # Add random jittering to samples (CRITICAL for anti-aliasing)


@dataclass
class GridConfig:
    """Occupancy grid configuration."""
    resolution: int = 128
    ema_tau: float = 0.95
    threshold: float = 0.01
    update_every: int = 16
    warmup_steps: int = 256
    levels: int = 1  # Single level for simplicity (was 4)
    use_grid: bool = True  # Actually use the grid for skipping


@dataclass
class PrecisionConfig:
    """Precision and mixed precision configuration."""
    param_dtype: str = "float32"
    compute_dtype: str = "float32"
    accum_dtype: str = "float32"
    use_amp: bool = False  # Disable AMP initially for stability


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_rays: int = 4096  # Batch size
    lr: float = 1e-2  # Base learning rate
    lr_encoder: float = 1e-2  # Higher LR for encoder (was 1e-2, should be higher)
    lr_mlp: float = 1e-3  # Separate LR for MLP (lower)
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1e-6
    eps: float = 1e-15
    iters: int = 20000
    seed: int = 1337
    deterministic: bool = False  # Disable for speed
    val_every: int = 1000
    log_every: int = 100
    use_lr_scheduler: bool = True  # Enable learning rate decay
    lr_decay_start: int = 1000  # When to start decaying
    lr_decay_end: int = 20000  # When to reach minimum LR
    lr_min_factor: float = 0.1  # Minimum LR as factor of initial


@dataclass
class Config:
    """Complete configuration bundle."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
