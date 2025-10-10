"""Performance microbenchmarks."""
import pytest
import torch
import time
from ngp_baseline_torch.config import Config
from ngp_baseline_torch.factory import create_all
from ngp_baseline_torch.rays import make_rays_single


@pytest.mark.perf
def test_perf_render_throughput(device):
    """Benchmark ray rendering throughput."""
    if device.type != 'cuda':
        pytest.skip("Performance test requires CUDA")

    cfg = Config()
    cfg.model.hash_levels = 16
    cfg.integrator.n_steps_fixed = 128

    encoder, field, rgb_head, _ = create_all(cfg, device)
    encoder.eval()
    field.eval()
    rgb_head.eval()

    # Generate rays
    H, W = 128, 128
    pose = torch.eye(4, device=device)
    rays = make_rays_single(H, W, pose, 400.0, 2.0, 6.0, device)

    # Warmup
    from ngp_baseline_torch.runtime import render_batch
    for _ in range(10):
        render_batch(rays, encoder, field, rgb_head, cfg)

    torch.cuda.synchronize()

    # Benchmark
    num_iters = 50
    start = time.time()

    for _ in range(num_iters):
        render_batch(rays, encoder, field, rgb_head, cfg)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    rays_per_sec = (rays.N * num_iters) / elapsed

    # Should achieve at least 1M rays/sec on modern GPU
    print(f"\nRay throughput: {rays_per_sec/1e6:.2f} M rays/s")
    assert rays_per_sec > 1e6, f"Performance too slow: {rays_per_sec/1e6:.2f} M rays/s"

