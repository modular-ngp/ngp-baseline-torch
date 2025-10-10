"""Data loading utilities for NeRF synthetic dataset."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def load_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """Load an image and convert to tensor.

    Args:
        image_path: Path to image file
        device: Target device

    Returns:
        Image tensor [H, W, 3] in range [0, 1]
    """
    img = Image.open(image_path)

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        if img.mode == 'RGBA':
            # Composite on white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        else:
            img = img.convert('RGB')

    # Convert to numpy array and normalize
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Convert to tensor
    img_tensor = torch.from_numpy(img_np).to(device)

    return img_tensor


def load_nerf_synthetic_images(scene_path: str | Path, split: str = "train",
                               device: torch.device = None) -> tuple[torch.Tensor, list]:
    """Load all images from NeRF synthetic dataset split.

    Args:
        scene_path: Path to scene directory
        split: Split name ("train", "val", "test")
        device: Target device, if None uses CPU

    Returns:
        images: Tensor [N, H, W, 3]
        image_paths: List of image paths
    """
    if device is None:
        device = torch.device('cpu')

    scene_path = Path(scene_path)

    # Determine transform file
    if split == "train":
        transform_file = "transforms_train.json"
    elif split == "val":
        transform_file = "transforms_val.json"
    elif split == "test":
        transform_file = "transforms_test.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load metadata
    with open(scene_path / transform_file, 'r') as f:
        meta = json.load(f)

    # Load images
    images = []
    image_paths = []

    for frame in meta['frames']:
        file_path = frame['file_path'].replace('./', '')
        img_path = scene_path / (file_path + '.png')

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        img = load_image(img_path, device)
        images.append(img)
        image_paths.append(img_path)

    # Stack images
    if images:
        images_tensor = torch.stack(images, dim=0)
    else:
        raise ValueError(f"No images found in {scene_path} for split {split}")

    return images_tensor, image_paths


class NeRFSyntheticDataset:
    """Dataset wrapper for NeRF synthetic scenes."""

    def __init__(self, scene_path: str | Path, split: str = "train", device: torch.device = None):
        """Initialize dataset.

        Args:
            scene_path: Path to scene directory
            split: Split name ("train", "val", "test")
            device: Target device
        """
        self.scene_path = Path(scene_path)
        self.split = split
        self.device = device or torch.device('cpu')

        # Load cameras
        from ..rays import load_nerf_synthetic

        transform_file = f"transforms_{split}.json"
        self.cameras = load_nerf_synthetic(self.scene_path, transform_file)
        self.cameras.poses = self.cameras.poses.to(self.device)

        # Load images lazily
        self._images = None
        self._image_paths = None

    @property
    def images(self) -> torch.Tensor:
        """Load images on demand."""
        if self._images is None:
            self._images, self._image_paths = load_nerf_synthetic_images(
                self.scene_path, self.split, self.device
            )
        return self._images

    @property
    def image_paths(self) -> list:
        """Get image paths."""
        if self._image_paths is None:
            _, self._image_paths = load_nerf_synthetic_images(
                self.scene_path, self.split, self.device
            )
        return self._image_paths

    def __len__(self) -> int:
        return self.cameras.N

    def __getitem__(self, idx: int) -> dict:
        """Get a single item.

        Returns:
            Dictionary with 'image', 'pose', 'idx'
        """
        return {
            'image': self.images[idx],
            'pose': self.cameras.poses[idx],
            'idx': idx
        }


def preload_dataset(scene: str = "lego", root: str = "data/nerf-synthetic",
                   splits: list[str] = None) -> dict:
    """Preload all dataset splits.

    Args:
        scene: Scene name
        root: Dataset root directory
        splits: List of splits to load, default is all

    Returns:
        Dictionary of datasets
    """
    if splits is None:
        splits = ["train", "val", "test"]

    scene_path = Path(root) / scene
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = {}
    for split in splits:
        print(f"Loading {split} split...")
        try:
            dataset = NeRFSyntheticDataset(scene_path, split, device)
            # Trigger image loading
            _ = dataset.images
            datasets[split] = dataset
            print(f"  Loaded {len(dataset)} images")
        except Exception as e:
            print(f"  Failed to load {split}: {e}")

    return datasets

