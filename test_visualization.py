"""
Quick test to verify the visualization module is working correctly.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from ngp_baseline_torch.visualization import NGPDebugServer
        print("  ✓ NGPDebugServer imported")
    except ImportError as e:
        print(f"  ✗ Failed to import NGPDebugServer: {e}")
        return False

    try:
        from ngp_baseline_torch.visualization import (
            sample_density_grid,
            filter_by_density_threshold,
            extract_training_metrics,
        )
        print("  ✓ Utility functions imported")
    except ImportError as e:
        print(f"  ✗ Failed to import utilities: {e}")
        return False

    return True


def test_server_creation():
    """Test server creation (without SHMX if not available)."""
    print("\nTesting server creation...")

    try:
        from ngp_baseline_torch.visualization import NGPDebugServer

        # Create server (but don't initialize if shmx not available)
        server = NGPDebugServer(
            name="test_server",
            max_points=10000,
        )
        print("  ✓ Server object created")

        # Check attributes
        assert server.name == "test_server"
        assert server.max_points == 10000
        print("  ✓ Server attributes correct")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_with_shmx():
    """Test actual server initialization (requires shmx)."""
    print("\nTesting with SHMX...")

    try:
        import shmx
        print("  ✓ shmx is available")
    except ImportError:
        print("  ⚠ shmx not installed (optional)")
        print("    Install with: pip install shmx")
        return None

    try:
        from ngp_baseline_torch.visualization import NGPDebugServer
        import numpy as np

        server = NGPDebugServer(name="test_ngp_server")

        if server.initialize():
            print("  ✓ Server initialized successfully")

            # Try publishing a frame
            positions = np.random.randn(100, 3).astype(np.float32)
            colors = np.random.rand(100, 3).astype(np.float32)
            densities = np.random.rand(100, 1).astype(np.float32)

            success = server.publish_frame(
                iteration=0,
                positions=positions,
                colors=colors,
                densities=densities,
                loss=0.1,
                psnr=25.0,
                learning_rate=1e-3,
            )

            if success:
                print("  ✓ Frame published successfully")
            else:
                print("  ✗ Failed to publish frame")

            server.shutdown()
            print("  ✓ Server shut down cleanly")

            return True
        else:
            print("  ✗ Failed to initialize server")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")

    try:
        from ngp_baseline_torch.visualization import filter_by_density_threshold
        import numpy as np

        positions = np.random.randn(1000, 3).astype(np.float32)
        colors = np.random.rand(1000, 3).astype(np.float32)
        densities = np.random.rand(1000, 1).astype(np.float32)

        # Test with numpy arrays
        pos_f, col_f, den_f = filter_by_density_threshold(
            positions, colors, densities,
            threshold=0.5,
            max_points=100,
        )

        assert pos_f.shape[0] <= 100
        assert pos_f.shape[0] == col_f.shape[0]
        assert pos_f.shape[0] == den_f.shape[0]
        print("  ✓ filter_by_density_threshold works with numpy")

        # Test with torch if available
        try:
            import torch
            positions_t = torch.from_numpy(positions)
            colors_t = torch.from_numpy(colors)
            densities_t = torch.from_numpy(densities)

            pos_f, col_f, den_f = filter_by_density_threshold(
                positions_t, colors_t, densities_t,
                threshold=0.5,
            )

            print("  ✓ filter_by_density_threshold works with torch")
        except ImportError:
            print("  ⚠ PyTorch not available for testing")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NGP Visualization Module - Quick Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Import test", test_imports()))
    results.append(("Server creation", test_server_creation()))
    results.append(("Utility functions", test_utility_functions()))
    results.append(("SHMX integration", test_with_shmx()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

    print("=" * 60)

