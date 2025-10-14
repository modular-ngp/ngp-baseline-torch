"""Test configuration dataclasses."""
import pytest
from ngp_baseline_torch.config import (
    Config, DatasetConfig, ModelConfig, IntegratorConfig,
    GridConfig, PrecisionConfig, TrainConfig
)


class TestConfig:
    """Test configuration classes."""

    def test_default_config(self):
        """Test default configuration creation."""
        cfg = Config()
        assert cfg.dataset.scene == "lego"
        assert cfg.model.hash_levels == 16
        assert cfg.train.batch_rays == 8192
        assert cfg.precision.use_amp is True

    def test_custom_config(self):
        """Test custom configuration."""
        cfg = Config()
        cfg.dataset.scene = "chair"
        cfg.model.mlp_width = 128
        cfg.train.lr = 0.001

        assert cfg.dataset.scene == "chair"
        assert cfg.model.mlp_width == 128
        assert cfg.train.lr == 0.001

    def test_dataset_config(self):
        """Test dataset configuration."""
        ds_cfg = DatasetConfig(scene="ficus", white_background=False)
        assert ds_cfg.scene == "ficus"
        assert ds_cfg.white_background is False

    def test_model_config(self):
        """Test model configuration."""
        model_cfg = ModelConfig(hash_levels=8, mlp_depth=3)
        assert model_cfg.hash_levels == 8
        assert model_cfg.mlp_depth == 3

