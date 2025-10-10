"""Comprehensive logging system for NGP training and debugging."""
from __future__ import annotations
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any
import torch
import torch.nn as nn
from .config import Config


class NGPLogger:
    """Logger for NGP training with comprehensive configuration and debug info."""

    def __init__(self, log_dir: str | Path, config: Config, run_name: str | None = None):
        """Initialize logger.

        Args:
            log_dir: Directory to save logs
            config: Complete configuration
            run_name: Optional run name, defaults to timestamp
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create run name with timestamp
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name

        # Set up file logging
        log_file = self.log_dir / f"{run_name}.log"
        self.logger = logging.getLogger(f"NGP_{run_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.config = config
        self.start_time = time.time()
        self.step_times = []

        # Log initialization
        self.logger.info("=" * 80)
        self.logger.info(f"NGP Training Run: {run_name}")
        self.logger.info("=" * 80)

    def log_system_info(self):
        """Log system and environment information."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 80)

        # PyTorch version
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"  GPU {i}: {props.name}")
                self.logger.info(f"    Memory: {props.total_memory / 1e9:.2f} GB")
                self.logger.info(f"    Compute Capability: {props.major}.{props.minor}")

        # Current device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Training Device: {device}")

    def log_complete_config(self):
        """Log all configuration parameters in detail."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPLETE CONFIGURATION")
        self.logger.info("=" * 80)

        cfg = self.config

        # Dataset Configuration
        self.logger.info("\n[DATASET CONFIGURATION]")
        self.logger.info(f"  Root Path: {cfg.dataset.root}")
        self.logger.info(f"  Scene: {cfg.dataset.scene}")
        self.logger.info(f"  Train Split: {cfg.dataset.train_split}")
        self.logger.info(f"  Val Split: {cfg.dataset.val_split}")
        self.logger.info(f"  Test Split: {cfg.dataset.test_split}")
        self.logger.info(f"  Scale: {cfg.dataset.scale}")
        self.logger.info(f"  AABB: {cfg.dataset.aabb}")

        # Model Configuration (Core NGP Parameters)
        self.logger.info("\n[MODEL ARCHITECTURE - NGP CORE]")
        self.logger.info("  Positional Encoding:")
        self.logger.info(f"    PE Bands (xyz): {cfg.model.pe_bands}")
        self.logger.info(f"    PE Bands (viewdir): {cfg.model.pe_viewdir_bands}")

        self.logger.info("  Hash Grid Encoding:")
        self.logger.info(f"    Levels: {cfg.model.hash_levels}")
        self.logger.info(f"    Base Resolution: {cfg.model.hash_res0}")
        self.logger.info(f"    Per-Level Scale: {cfg.model.hash_per_level_scale}")
        self.logger.info(f"    Features per Level: {cfg.model.hash_features_per_level}")

        # Calculate max resolution and feature dimensions
        if cfg.model.hash_levels > 0:
            max_res = int(cfg.model.hash_res0 * (cfg.model.hash_per_level_scale ** (cfg.model.hash_levels - 1)))
            total_hash_features = cfg.model.hash_levels * cfg.model.hash_features_per_level
            self.logger.info(f"    Computed Max Resolution: {max_res}")
            self.logger.info(f"    Total Hash Features: {total_hash_features}")

        self.logger.info("  MLP Architecture:")
        self.logger.info(f"    Hidden Dimension: {cfg.model.mlp_width}")
        self.logger.info(f"    Number of Layers: {cfg.model.mlp_depth}")
        self.logger.info(f"    Activation: {cfg.model.activation}")
        self.logger.info(f"    View Dependent: {cfg.model.view_dependent}")

        # Volume Integration Configuration
        self.logger.info("\n[VOLUME INTEGRATION]")
        self.logger.info(f"  Step Strategy: {cfg.integrator.step_strategy}")
        self.logger.info(f"  Fixed Steps per Ray: {cfg.integrator.n_steps_fixed}")
        self.logger.info(f"  Sigma Threshold: {cfg.integrator.sigma_thresh}")
        self.logger.info(f"  Early Stop Transmittance: {cfg.integrator.early_stop_T}")
        self.logger.info(f"  DT Gamma (adaptive): {cfg.integrator.dt_gamma}")

        # Occupancy Grid Configuration
        self.logger.info("\n[OCCUPANCY GRID]")
        self.logger.info(f"  Resolution: {cfg.grid.resolution}")
        self.logger.info(f"  EMA Tau: {cfg.grid.ema_tau}")
        self.logger.info(f"  Occupancy Threshold: {cfg.grid.threshold}")
        self.logger.info(f"  Update Every N Steps: {cfg.grid.update_every}")
        self.logger.info(f"  Warmup Steps: {cfg.grid.warmup_steps}")
        self.logger.info(f"  Levels: {cfg.grid.levels}")

        # Precision Configuration
        self.logger.info("\n[PRECISION & MIXED PRECISION]")
        self.logger.info(f"  Parameter Dtype: {cfg.precision.param_dtype}")
        self.logger.info(f"  Compute Dtype: {cfg.precision.compute_dtype}")
        self.logger.info(f"  Accumulator Dtype: {cfg.precision.accum_dtype}")
        self.logger.info(f"  Use AMP: {cfg.precision.use_amp}")

        # Training Configuration
        self.logger.info("\n[TRAINING HYPERPARAMETERS]")
        self.logger.info(f"  Batch Size (rays): {cfg.train.batch_rays}")
        self.logger.info(f"  Learning Rate: {cfg.train.lr}")
        self.logger.info(f"  Optimizer Betas: {cfg.train.betas}")
        self.logger.info(f"  Weight Decay: {cfg.train.weight_decay}")
        self.logger.info(f"  Epsilon: {cfg.train.eps}")
        self.logger.info(f"  Total Iterations: {cfg.train.iters}")
        self.logger.info(f"  Random Seed: {cfg.train.seed}")
        self.logger.info(f"  Deterministic Mode: {cfg.train.deterministic}")
        self.logger.info(f"  Validation Every: {cfg.train.val_every} steps")
        self.logger.info(f"  Logging Every: {cfg.train.log_every} steps")

        # Save config as JSON
        config_file = self.log_dir / f"{self.run_name}_config.json"
        self._save_config_json(config_file)
        self.logger.info(f"\nConfiguration saved to: {config_file}")

    def _save_config_json(self, path: Path):
        """Save configuration as JSON file."""
        config_dict = {
            'dataset': {
                'root': self.config.dataset.root,
                'scene': self.config.dataset.scene,
                'train_split': self.config.dataset.train_split,
                'val_split': self.config.dataset.val_split,
                'test_split': self.config.dataset.test_split,
                'scale': self.config.dataset.scale,
                'aabb': self.config.dataset.aabb,
            },
            'model': {
                'pe_bands': self.config.model.pe_bands,
                'hash_levels': self.config.model.hash_levels,
                'hash_res0': self.config.model.hash_res0,
                'hash_per_level_scale': self.config.model.hash_per_level_scale,
                'hash_features_per_level': self.config.model.hash_features_per_level,
                'mlp_width': self.config.model.mlp_width,
                'mlp_depth': self.config.model.mlp_depth,
                'activation': self.config.model.activation,
                'view_dependent': self.config.model.view_dependent,
                'pe_viewdir_bands': self.config.model.pe_viewdir_bands,
            },
            'integrator': {
                'step_strategy': self.config.integrator.step_strategy,
                'n_steps_fixed': self.config.integrator.n_steps_fixed,
                'sigma_thresh': self.config.integrator.sigma_thresh,
                'early_stop_T': self.config.integrator.early_stop_T,
                'dt_gamma': self.config.integrator.dt_gamma,
            },
            'grid': {
                'resolution': self.config.grid.resolution,
                'ema_tau': self.config.grid.ema_tau,
                'threshold': self.config.grid.threshold,
                'update_every': self.config.grid.update_every,
                'warmup_steps': self.config.grid.warmup_steps,
                'levels': self.config.grid.levels,
            },
            'precision': {
                'param_dtype': self.config.precision.param_dtype,
                'compute_dtype': self.config.precision.compute_dtype,
                'accum_dtype': self.config.precision.accum_dtype,
                'use_amp': self.config.precision.use_amp,
            },
            'train': {
                'batch_rays': self.config.train.batch_rays,
                'lr': self.config.train.lr,
                'betas': self.config.train.betas,
                'weight_decay': self.config.train.weight_decay,
                'eps': self.config.train.eps,
                'iters': self.config.train.iters,
                'seed': self.config.train.seed,
                'deterministic': self.config.train.deterministic,
                'val_every': self.config.train.val_every,
                'log_every': self.config.train.log_every,
            }
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def log_model_info(self, encoder: nn.Module, field: nn.Module, rgb_head: nn.Module,
                       occupancy_grid: nn.Module | None = None):
        """Log detailed model architecture and parameter counts."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MODEL STATISTICS")
        self.logger.info("=" * 80)

        # Encoder
        encoder_params = sum(p.numel() for p in encoder.parameters())
        encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        self.logger.info(f"\n[ENCODER]")
        self.logger.info(f"  Type: {encoder.__class__.__name__}")
        self.logger.info(f"  Total Parameters: {encoder_params:,}")
        self.logger.info(f"  Trainable Parameters: {encoder_trainable:,}")
        if hasattr(encoder, 'output_dim'):
            self.logger.info(f"  Output Dimension: {encoder.output_dim}")

        # Field MLP
        field_params = sum(p.numel() for p in field.parameters())
        field_trainable = sum(p.numel() for p in field.parameters() if p.requires_grad)
        self.logger.info(f"\n[FIELD MLP]")
        self.logger.info(f"  Type: {field.__class__.__name__}")
        self.logger.info(f"  Total Parameters: {field_params:,}")
        self.logger.info(f"  Trainable Parameters: {field_trainable:,}")

        # RGB Head
        rgb_params = sum(p.numel() for p in rgb_head.parameters())
        rgb_trainable = sum(p.numel() for p in rgb_head.parameters() if p.requires_grad)
        self.logger.info(f"\n[RGB HEAD]")
        self.logger.info(f"  Type: {rgb_head.__class__.__name__}")
        self.logger.info(f"  Total Parameters: {rgb_params:,}")
        self.logger.info(f"  Trainable Parameters: {rgb_trainable:,}")

        # Occupancy Grid
        if occupancy_grid is not None:
            grid_params = sum(p.numel() for p in occupancy_grid.parameters())
            self.logger.info(f"\n[OCCUPANCY GRID]")
            self.logger.info(f"  Type: {occupancy_grid.__class__.__name__}")
            self.logger.info(f"  Total Parameters: {grid_params:,}")

        # Total
        total_params = encoder_params + field_params + rgb_params
        total_trainable = encoder_trainable + field_trainable + rgb_trainable
        self.logger.info(f"\n[TOTAL]")
        self.logger.info(f"  Total Parameters: {total_params:,}")
        self.logger.info(f"  Trainable Parameters: {total_trainable:,}")

        # Memory estimate
        param_memory_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        self.logger.info(f"  Estimated Parameter Memory: {param_memory_mb:.2f} MB")

    def log_dataset_info(self, num_images: int, image_shape: tuple, num_rays: int):
        """Log dataset statistics."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("DATASET STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"  Number of Training Images: {num_images}")
        self.logger.info(f"  Image Resolution: {image_shape[0]}x{image_shape[1]}")
        self.logger.info(f"  Total Training Rays: {num_rays:,}")
        self.logger.info(f"  Rays per Image: {num_rays // num_images:,}")

    def log_training_start(self):
        """Log training start."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING START")
        self.logger.info("=" * 80)
        self.start_time = time.time()

    def log_step(self, step: int, metrics: dict[str, Any], lr: float | None = None):
        """Log training step metrics.

        Args:
            step: Current training step
            metrics: Dictionary of metrics (loss, psnr, etc.)
            lr: Current learning rate
        """
        # Update step times
        if len(self.step_times) > 0:
            elapsed = time.time() - self.step_times[-1]
        else:
            elapsed = 0
        self.step_times.append(time.time())

        # Format metrics
        metric_str = " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items()])

        if lr is not None:
            metric_str += f" | lr: {lr:.6e}"

        if elapsed > 0:
            metric_str += f" | time: {elapsed:.3f}s"

        self.logger.info(f"Step {step:6d} | {metric_str}")

    def log_validation(self, step: int, val_metrics: dict[str, Any]):
        """Log validation metrics."""
        metric_str = " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in val_metrics.items()])
        self.logger.info(f"Validation @ Step {step:6d} | {metric_str}")

    def log_epoch_summary(self, start_step: int, end_step: int, avg_metrics: dict[str, float]):
        """Log summary statistics for a range of steps."""
        self.logger.info("\n" + "-" * 80)
        self.logger.info(f"Summary: Steps {start_step}-{end_step}")

        for k, v in avg_metrics.items():
            self.logger.info(f"  Average {k}: {v:.6f}")

        # Time statistics
        if len(self.step_times) > 1:
            recent_times = self.step_times[-100:]
            if len(recent_times) > 1:
                avg_time = (recent_times[-1] - recent_times[0]) / (len(recent_times) - 1)
                self.logger.info(f"  Average Step Time: {avg_time:.3f}s")
                self.logger.info(f"  Steps/Second: {1.0/avg_time:.2f}")

        self.logger.info("-" * 80 + "\n")

    def log_gradient_stats(self, encoder: nn.Module, field: nn.Module, rgb_head: nn.Module):
        """Log gradient statistics for debugging."""
        self.logger.debug("\n[GRADIENT STATISTICS]")

        modules = [("Encoder", encoder), ("Field", field), ("RGB Head", rgb_head)]

        for name, module in modules:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if len(grads) > 0:
                grad_norms = [g.norm().item() for g in grads]
                self.logger.debug(f"  {name}:")
                self.logger.debug(f"    Mean Grad Norm: {sum(grad_norms)/len(grad_norms):.6e}")
                self.logger.debug(f"    Max Grad Norm: {max(grad_norms):.6e}")
                self.logger.debug(f"    Min Grad Norm: {min(grad_norms):.6e}")

    def log_weight_stats(self, encoder: nn.Module, field: nn.Module, rgb_head: nn.Module):
        """Log weight statistics for debugging."""
        self.logger.debug("\n[WEIGHT STATISTICS]")

        modules = [("Encoder", encoder), ("Field", field), ("RGB Head", rgb_head)]

        for name, module in modules:
            weights = [p for p in module.parameters() if p.requires_grad]
            if len(weights) > 0:
                weight_norms = [w.norm().item() for w in weights]
                weight_means = [w.mean().item() for w in weights]
                weight_stds = [w.std().item() for w in weights]

                self.logger.debug(f"  {name}:")
                self.logger.debug(f"    Mean Weight Norm: {sum(weight_norms)/len(weight_norms):.6e}")
                self.logger.debug(f"    Mean Weight Mean: {sum(weight_means)/len(weight_means):.6e}")
                self.logger.debug(f"    Mean Weight Std: {sum(weight_stds)/len(weight_stds):.6e}")

    def log_rendering_stats(self, aux: dict[str, Any]):
        """Log rendering-specific statistics."""
        self.logger.debug("[RENDERING STATISTICS]")
        for k, v in aux.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    self.logger.debug(f"  {k}: {v.item():.6f}")
                else:
                    self.logger.debug(f"  {k}: mean={v.mean().item():.6f}, std={v.std().item():.6f}")
            else:
                self.logger.debug(f"  {k}: {v}")

    def log_training_complete(self, final_step: int, final_metrics: dict[str, float]):
        """Log training completion."""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"  Final Step: {final_step}")
        self.logger.info(f"  Total Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.logger.info(f"  Average Time per Step: {total_time/final_step:.3f}s")

        self.logger.info("\n[FINAL METRICS]")
        for k, v in final_metrics.items():
            self.logger.info(f"  {k}: {v:.6f}")

        self.logger.info("=" * 80)

    def log_debug_batch(self, rays, target_rgb, pred_rgb, aux):
        """Log detailed debug information for a single batch."""
        self.logger.debug("\n[BATCH DEBUG INFO]")
        self.logger.debug(f"  Ray Count: {rays.N}")
        self.logger.debug(f"  Target RGB: min={target_rgb.min().item():.3f}, max={target_rgb.max().item():.3f}, mean={target_rgb.mean().item():.3f}")
        self.logger.debug(f"  Pred RGB: min={pred_rgb.min().item():.3f}, max={pred_rgb.max().item():.3f}, mean={pred_rgb.mean().item():.3f}")
        self.logger.debug(f"  Ray tmin: min={rays.tmin.min().item():.3f}, max={rays.tmin.max().item():.3f}")
        self.logger.debug(f"  Ray tmax: min={rays.tmax.min().item():.3f}, max={rays.tmax.max().item():.3f}")

        if 'sigma' in aux:
            sigma = aux['sigma']
            self.logger.debug(f"  Sigma: min={sigma.min().item():.3f}, max={sigma.max().item():.3f}, mean={sigma.mean().item():.3f}")

        if 'transmittance' in aux:
            T = aux['transmittance']
            self.logger.debug(f"  Transmittance: min={T.min().item():.3f}, max={T.max().item():.3f}, mean={T.mean().item():.3f}")

