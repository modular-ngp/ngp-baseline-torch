"""
Simple standalone example of NGP Debug Server usage.

This script demonstrates how to use the NGPDebugServer to publish
debug data to shared memory for visualization.
"""

import numpy as np
import time
from ngp_baseline_torch.visualization import NGPDebugServer


def example_basic_usage():
    """Basic usage example of NGPDebugServer."""

    print("=" * 60)
    print("NGP Debug Server - Basic Usage Example")
    print("=" * 60)

    # Create and initialize server
    server = NGPDebugServer(
        name="ngp_example",
        max_points=100_000,
        slots=4,
        reader_slots=8,
    )

    if not server.initialize():
        print("Failed to initialize server. Is shmx installed?")
        print("Install with: pip install shmx")
        return

    print("\nServer initialized successfully!")
    print("You can now connect a visualization client to 'ngp_example'")
    print("\nPublishing sample data for 10 seconds...\n")

    # Publish sample data
    for iteration in range(100):
        # Generate sample data
        num_points = 10000

        # Random 3D positions in a sphere
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        r = np.random.uniform(0.2, 1.0, num_points)

        positions = np.stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ], axis=-1).astype(np.float32)

        # Color based on position (normalized)
        colors = (positions + 1.0) / 2.0
        colors = np.clip(colors, 0, 1).astype(np.float32)

        # Density based on distance from center
        densities = (1.0 - np.linalg.norm(positions, axis=-1, keepdims=True)).astype(np.float32)
        densities = np.clip(densities, 0, 1)

        # Simulated training metrics
        loss = 0.1 * np.exp(-iteration / 20.0) + 0.01
        psnr = 15.0 + 10.0 * (1.0 - np.exp(-iteration / 30.0))
        learning_rate = 1e-3 * np.exp(-iteration / 50.0)

        # Camera position (rotating)
        angle = iteration * 0.1
        camera_pos = np.array([
            3.0 * np.cos(angle),
            3.0 * np.sin(angle),
            1.5
        ], dtype=np.float32)

        camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Publish frame
        success = server.publish_frame(
            iteration=iteration,
            positions=positions,
            colors=colors,
            densities=densities,
            loss=loss,
            psnr=psnr,
            learning_rate=learning_rate,
            camera_pos=camera_pos,
            camera_target=camera_target,
        )

        if success and iteration % 10 == 0:
            print(f"Iter {iteration:3d}: Published {num_points:,} points | "
                  f"Loss: {loss:.4f} | PSNR: {psnr:.2f} dB | "
                  f"Frame #{server.frame_count}")

        # Sleep to simulate training
        time.sleep(0.1)

    print("\n" + "=" * 60)
    print("Example complete!")
    print(f"Total frames published: {server.frame_count}")
    print("=" * 60)

    # Cleanup
    server.shutdown()


def example_with_torch():
    """Example using PyTorch tensors."""

    try:
        import torch
    except ImportError:
        print("PyTorch not available. Skipping torch example.")
        return

    print("\n" + "=" * 60)
    print("NGP Debug Server - PyTorch Integration Example")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create server
    server = NGPDebugServer(name="ngp_torch_example")

    if not server.initialize():
        print("Failed to initialize server.")
        return

    print("\nPublishing PyTorch tensor data...\n")

    for iteration in range(50):
        # Generate torch tensors
        num_points = 20000
        positions = torch.randn(num_points, 3, device=device) * 0.5
        colors = torch.rand(num_points, 3, device=device)
        densities = torch.exp(-torch.norm(positions, dim=-1, keepdim=True))

        # Publish (automatically handles GPU->CPU transfer)
        server.publish_frame(
            iteration=iteration,
            positions=positions,
            colors=colors,
            densities=densities,
            loss=0.05,
            psnr=28.5,
            learning_rate=1e-3,
        )

        if iteration % 10 == 0:
            print(f"Iter {iteration:3d}: Published frame #{server.frame_count}")

        time.sleep(0.05)

    print(f"\nPublished {server.frame_count} frames with PyTorch tensors")
    server.shutdown()


def example_context_manager():
    """Example using context manager syntax."""

    print("\n" + "=" * 60)
    print("NGP Debug Server - Context Manager Example")
    print("=" * 60)

    # Use context manager for automatic cleanup
    with NGPDebugServer(name="ngp_context_example") as server:
        print("\nServer created with context manager")

        for i in range(20):
            positions = np.random.randn(5000, 3).astype(np.float32) * 0.5
            colors = np.random.rand(5000, 3).astype(np.float32)
            densities = np.random.rand(5000, 1).astype(np.float32)

            server.publish_frame(
                iteration=i,
                positions=positions,
                colors=colors,
                densities=densities,
            )

            time.sleep(0.05)

        print(f"Published {server.frame_count} frames")

    print("Server automatically shut down on context exit")


if __name__ == "__main__":
    import sys

    print("\nNGP Debug Server Examples")
    print("=" * 60)
    print("Make sure 'shmx' is installed: pip install shmx")
    print("=" * 60)

    try:
        # Run examples
        example_basic_usage()

        if '--torch' in sys.argv:
            example_with_torch()

        if '--context' in sys.argv:
            example_context_manager()

        print("\n✓ All examples completed successfully!")
        print("\nTip: Run your Vulkan visualizer to connect and see the data.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

