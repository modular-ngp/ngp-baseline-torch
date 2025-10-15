"""
NGP Debug Server using SHMX for zero-copy IPC visualization.

This module provides a server that publishes Instant NGP training debug data
to shared memory for real-time visualization by external processes.
"""

from __future__ import annotations
import time
from typing import Optional, Dict, Any
import numpy as np

try:
    import shmx
    SHMX_AVAILABLE = True
except ImportError:
    SHMX_AVAILABLE = False
    print("[NGPDebugServer] Warning: shmx not available. Install with: pip install shmx")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NGPDebugServer:
    """
    Instant NGP Debug Server for real-time visualization via shared memory.

    Publishes training data (point clouds, densities, colors, metrics) to shared memory
    for consumption by external Vulkan/visualization processes.
    """

    # Stream ID definitions
    STREAM_FRAME_SEQ = 1
    STREAM_TIMESTAMP = 2
    STREAM_ITERATION = 3
    STREAM_POSITIONS = 100
    STREAM_COLORS = 101
    STREAM_NORMALS = 102
    STREAM_DENSITY = 200
    STREAM_OPACITY = 201
    STREAM_FEATURES = 202
    STREAM_LOSS = 300
    STREAM_PSNR = 301
    STREAM_LEARNING_RATE = 302
    STREAM_CAMERA_POS = 400
    STREAM_CAMERA_TARGET = 401
    STREAM_CAMERA_MATRIX = 402

    def __init__(
        self,
        name: str = "ngp_debug",
        max_points: int = 500_000,
        max_rays: int = 4096,
        enable_volume: bool = False,
        volume_resolution: int = 128,
        slots: int = 4,
        reader_slots: int = 16,
    ):
        """
        Initialize NGP Debug Server.

        Args:
            name: Shared memory region name
            max_points: Maximum number of 3D points per frame
            max_rays: Maximum number of rays per frame
            enable_volume: Whether to support volume data
            volume_resolution: Resolution for volume data (e.g., 128^3)
            slots: Number of ring buffer slots
            reader_slots: Maximum number of concurrent clients
        """
        if not SHMX_AVAILABLE:
            raise RuntimeError("shmx library not available. Install with: pip install shmx")

        self.name = name
        self.max_points = max_points
        self.max_rays = max_rays
        self.enable_volume = enable_volume
        self.volume_resolution = volume_resolution
        self.slots = slots
        self.reader_slots = reader_slots

        self.server: Optional[Any] = None
        self.frame_count = 0
        self.enabled = True

        # Calculate required buffer size
        self.frame_bytes_cap = self._calculate_frame_size()

    def _calculate_frame_size(self) -> int:
        """Calculate maximum frame data size in bytes."""
        size = 0

        # Metadata (~1KB)
        size += 1024

        # Point cloud data
        size += self.max_points * 3 * 4  # positions (float32 x 3)
        size += self.max_points * 3 * 4  # colors (float32 x 3)
        size += self.max_points * 1 * 4  # densities (float32 x 1)
        size += self.max_points * 3 * 4  # normals (optional, float32 x 3)

        # Ray samples
        size += self.max_rays * 3 * 4    # ray origins
        size += self.max_rays * 3 * 4    # ray directions

        # Training stats
        size += 1024  # Various scalars

        # Volume data (optional)
        if self.enable_volume:
            voxels = self.volume_resolution ** 3
            size += voxels * 1 * 4  # density volume
            size += voxels * 3 * 4  # color volume

        # Add 20% overhead for TLV headers
        size = int(size * 1.2)

        return size

    def initialize(self) -> bool:
        """
        Initialize shared memory server with stream definitions.

        Returns:
            True if successful, False otherwise
        """
        if not SHMX_AVAILABLE:
            return False

        try:
            # Define stream specifications
            streams = self._create_stream_specs()

            # Create server
            self.server = shmx.Server()
            success = self.server.create(
                name=self.name,
                slots=self.slots,
                reader_slots=self.reader_slots,
                static_bytes_cap=8192,
                frame_bytes_cap=self.frame_bytes_cap,
                control_per_reader=4096,
                streams=streams
            )

            if success:
                header_info = self.server.get_header_info()
                print(f"[NGPDebugServer] Initialized: '{self.name}'")
                print(f"  Frame buffer size: {self.frame_bytes_cap / 1024 / 1024:.2f} MB")
                print(f"  Session ID: {header_info.get('session_id', 'N/A')}")
                print(f"  Slots: {self.slots}, Reader slots: {self.reader_slots}")
            else:
                print(f"[NGPDebugServer] Failed to initialize server!")

            return success

        except Exception as e:
            print(f"[NGPDebugServer] Error during initialization: {e}")
            return False

    def _create_stream_specs(self) -> list:
        """Create stream specification list for SHMX."""
        streams = [
            # === Metadata Streams ===
            {
                'id': self.STREAM_FRAME_SEQ,
                'name': "frame_seq",
                'dtype_code': shmx.DT_U64,
                'components': 1,
                'bytes_per_elem': 8
            },
            {
                'id': self.STREAM_TIMESTAMP,
                'name': "timestamp",
                'dtype_code': shmx.DT_F64,
                'components': 1,
                'bytes_per_elem': 8
            },
            {
                'id': self.STREAM_ITERATION,
                'name': "iteration",
                'dtype_code': shmx.DT_U32,
                'components': 1,
                'bytes_per_elem': 4
            },

            # === Geometry Streams ===
            {
                'id': self.STREAM_POSITIONS,
                'name': "positions",
                'dtype_code': shmx.DT_F32,
                'components': 3,
                'bytes_per_elem': 12,
                'layout_code': shmx.LAYOUT_AOS_VECTOR
            },
            {
                'id': self.STREAM_COLORS,
                'name': "colors",
                'dtype_code': shmx.DT_F32,
                'components': 3,
                'bytes_per_elem': 12,
                'layout_code': shmx.LAYOUT_AOS_VECTOR
            },
            {
                'id': self.STREAM_NORMALS,
                'name': "normals",
                'dtype_code': shmx.DT_F32,
                'components': 3,
                'bytes_per_elem': 12,
                'layout_code': shmx.LAYOUT_AOS_VECTOR
            },

            # === Field Data Streams ===
            {
                'id': self.STREAM_DENSITY,
                'name': "density",
                'dtype_code': shmx.DT_F32,
                'components': 1,
                'bytes_per_elem': 4
            },
            {
                'id': self.STREAM_OPACITY,
                'name': "opacity",
                'dtype_code': shmx.DT_F32,
                'components': 1,
                'bytes_per_elem': 4
            },

            # === Training Stats Streams ===
            {
                'id': self.STREAM_LOSS,
                'name': "loss",
                'dtype_code': shmx.DT_F32,
                'components': 1,
                'bytes_per_elem': 4
            },
            {
                'id': self.STREAM_PSNR,
                'name': "psnr",
                'dtype_code': shmx.DT_F32,
                'components': 1,
                'bytes_per_elem': 4
            },
            {
                'id': self.STREAM_LEARNING_RATE,
                'name': "learning_rate",
                'dtype_code': shmx.DT_F32,
                'components': 1,
                'bytes_per_elem': 4
            },

            # === Camera Streams ===
            {
                'id': self.STREAM_CAMERA_POS,
                'name': "camera_pos",
                'dtype_code': shmx.DT_F32,
                'components': 3,
                'bytes_per_elem': 12
            },
            {
                'id': self.STREAM_CAMERA_TARGET,
                'name': "camera_target",
                'dtype_code': shmx.DT_F32,
                'components': 3,
                'bytes_per_elem': 12
            },
            {
                'id': self.STREAM_CAMERA_MATRIX,
                'name': "camera_matrix",
                'dtype_code': shmx.DT_F32,
                'components': 16,
                'bytes_per_elem': 64
            },
        ]

        return streams

    def publish_frame(
        self,
        iteration: int,
        positions: Optional[Any] = None,
        colors: Optional[Any] = None,
        densities: Optional[Any] = None,
        normals: Optional[Any] = None,
        loss: Optional[float] = None,
        psnr: Optional[float] = None,
        learning_rate: Optional[float] = None,
        camera_pos: Optional[np.ndarray] = None,
        camera_target: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None,
        **kwargs
    ) -> bool:
        """
        Publish a frame of debug data to shared memory.

        Args:
            iteration: Current training iteration
            positions: [N, 3] 3D positions (torch.Tensor or numpy array)
            colors: [N, 3] RGB colors (torch.Tensor or numpy array)
            densities: [N,] or [N, 1] density values (torch.Tensor or numpy array)
            normals: [N, 3] normal vectors (optional)
            loss: Current loss value
            psnr: Current PSNR metric
            learning_rate: Current learning rate
            camera_pos: [3,] camera position
            camera_target: [3,] camera target point
            camera_matrix: [4, 4] or [16,] camera transformation matrix
            **kwargs: Additional custom data

        Returns:
            True if published successfully, False otherwise
        """
        if not self.enabled or self.server is None:
            return False

        try:
            # Begin new frame
            frame_handle = self.server.begin_frame()
            timestamp = time.time()

            # === Metadata ===
            self._append_scalar(frame_handle, self.STREAM_FRAME_SEQ, self.frame_count, np.uint64)
            self._append_scalar(frame_handle, self.STREAM_TIMESTAMP, timestamp, np.float64)
            self._append_scalar(frame_handle, self.STREAM_ITERATION, iteration, np.uint32)

            # === Geometry Data ===
            if positions is not None:
                self._append_tensor(frame_handle, self.STREAM_POSITIONS, positions)

            if colors is not None:
                self._append_tensor(frame_handle, self.STREAM_COLORS, colors)

            if normals is not None:
                self._append_tensor(frame_handle, self.STREAM_NORMALS, normals)

            # === Field Data ===
            if densities is not None:
                self._append_tensor(frame_handle, self.STREAM_DENSITY, densities)

            # === Training Stats ===
            if loss is not None:
                self._append_scalar(frame_handle, self.STREAM_LOSS, loss, np.float32)

            if psnr is not None:
                self._append_scalar(frame_handle, self.STREAM_PSNR, psnr, np.float32)

            if learning_rate is not None:
                self._append_scalar(frame_handle, self.STREAM_LEARNING_RATE, learning_rate, np.float32)

            # === Camera Data ===
            if camera_pos is not None:
                data = np.asarray(camera_pos, dtype=np.float32).flatten()
                if data.size == 3:
                    self.server.append_stream(frame_handle, self.STREAM_CAMERA_POS, data.tobytes(), 1)

            if camera_target is not None:
                data = np.asarray(camera_target, dtype=np.float32).flatten()
                if data.size == 3:
                    self.server.append_stream(frame_handle, self.STREAM_CAMERA_TARGET, data.tobytes(), 1)

            if camera_matrix is not None:
                data = np.asarray(camera_matrix, dtype=np.float32).flatten()
                if data.size == 16:
                    self.server.append_stream(frame_handle, self.STREAM_CAMERA_MATRIX, data.tobytes(), 1)

            # Publish frame
            success = self.server.publish_frame(frame_handle, timestamp)

            if success:
                self.frame_count += 1

            return success

        except Exception as e:
            print(f"[NGPDebugServer] Error publishing frame: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _append_scalar(self, frame_handle, stream_id: int, value, dtype):
        """Append scalar data stream."""
        data = np.array([value], dtype=dtype)
        self.server.append_stream(
            frame_handle,
            stream_id,
            data.tobytes(),
            1  # elem_count
        )

    def _append_tensor(self, frame_handle, stream_id: int, tensor):
        """
        Append tensor data stream with zero-copy from torch.Tensor or numpy array.

        Args:
            frame_handle: Frame handle from begin_frame()
            stream_id: Stream ID
            tensor: torch.Tensor or numpy array
        """
        # Convert to numpy if torch tensor
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            # Move to CPU if on GPU
            if tensor.is_cuda:
                tensor = tensor.cpu()
            # Ensure contiguous
            tensor = tensor.contiguous()
            # Convert to float32 if needed
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            # Convert to numpy
            arr = tensor.numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32)

        # Ensure contiguous
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)

        # Handle different shapes
        if arr.ndim == 1:
            num_elements = arr.shape[0]
        elif arr.ndim == 2:
            num_elements = arr.shape[0]
        else:
            # Flatten to 2D
            arr = arr.reshape(-1, arr.shape[-1])
            num_elements = arr.shape[0]

        # Append to stream
        data_bytes = arr.tobytes()
        self.server.append_stream(
            frame_handle,
            stream_id,
            data_bytes,
            num_elements
        )

    def poll_control_messages(self, max_messages: int = 256) -> list:
        """
        Poll control messages from clients.

        Args:
            max_messages: Maximum number of messages to retrieve

        Returns:
            List of control message dictionaries
        """
        if self.server is None:
            return []

        try:
            return self.server.poll_control(max_messages)
        except Exception as e:
            print(f"[NGPDebugServer] Error polling control messages: {e}")
            return []

    def set_enabled(self, enabled: bool):
        """Enable or disable publishing (for performance)."""
        self.enabled = enabled

    def is_enabled(self) -> bool:
        """Check if publishing is enabled."""
        return self.enabled

    def shutdown(self):
        """Shutdown the server and release shared memory."""
        if self.server is not None:
            try:
                self.server.destroy()
                print(f"[NGPDebugServer] Shutdown complete. Published {self.frame_count} frames.")
            except Exception as e:
                print(f"[NGPDebugServer] Error during shutdown: {e}")
            finally:
                self.server = None

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

    def __del__(self):
        """Destructor."""
        self.shutdown()
"""Visualization module for real-time NGP debugging via shared memory."""

from .debug_server import NGPDebugServer

__all__ = ['NGPDebugServer']
