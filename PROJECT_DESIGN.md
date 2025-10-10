# NGP-Baseline-Torch — Project Design (for AI agent execution)

Status: v1.0
Audience: AI agent implementing a strictly decoupled, minimal-yet-correct Instant-NGP PyTorch baseline with exhaustive tests.
Dataset assumption: `data/nerf-synthetic/` exists at project root; only this dataset is supported in v1.0.

---

## 0) Design goals

* Decoupled: each module is Simple-In Simple-Out (SISO); no hidden globals; replaceable later by CUDA.
* Minimal: include only modules strictly required by Instant-NGP-style NeRF; no extras.
* Correct: numerically sound forward/backward; reproducible with fixed seeds.
* Modern: `pyproject.toml`, torch AMP, device/dtype policy centralized.
* Test-first: exhaustive unit tests under `tests/` cover interfaces, numerics, grads, determinism, and baseline performance.

Non-goals: CLI apps, viewers, docs beyond this file, multi-dataset support, complex configs.

---

## 1) Repository layout (minimal, decoupled)

```
ngp-baseline-torch/
  pyproject.toml
  .gitignore

  src/
    ngp_baseline_torch/
      __init__.py

      types.py
      config.py
      device.py
      rng.py
      factory.py

      rays/
        __init__.py
        cameras.py
        rays.py

      encoder/
        __init__.py
        pe.py
        hashgrid_torch.py

      field/
        __init__.py
        mlp.py
        heads.py

      integrator/
        __init__.py
        marcher.py
        compositor.py

      grid/
        __init__.py
        occupancy.py

      loss/
        __init__.py
        rgb.py

      opt/
        __init__.py
        adam.py

      runtime/
        __init__.py
        train.py
        infer.py

      artifact/
        __init__.py
        export_v0.py

  tests/
    conftest.py
    test_types_and_config.py
    test_rng_and_device.py
    test_rays_and_cameras.py
    test_encoder_pe.py
    test_encoder_hashgrid.py
    test_field_mlp_and_heads.py
    test_integrator_compositor_analytic.py
    test_integrator_training_loop_sanity.py
    test_grid_occupancy.py
    test_loss_rgb.py
    test_opt_adam.py
    test_runtime_train_step.py
    test_artifact_export_v0.py
    test_end_to_end_render_psnr_quick.py
    test_grad_check_small_batch.py
    test_reproducibility.py
    test_perf_microbench.py
```

Notes:

* All implementation code lives under `src/ngp_baseline_torch/`.
* All tests live under `tests/` and are exhaustive (interfaces, numerics, grads, perf).
* No other assets or scripts are included.

---

## 2) Contract layer (SISO)

### 2.1 `types.py`

Define explicit, framework-agnostic types and shape invariants (documented here, enforced by asserts in code):

* RayBatch (SoA on device):

    * orig_x[N], orig_y[N], orig_z[N] : float32
    * dir_x[N],  dir_y[N],  dir_z[N]  : float32
    * tmin[N], tmax[N] : float32
    * mask[N] : bool (optional)
    * N arbitrary; all arrays must share device and dtype; SoA length must match.

* EncodedFeat:

    * feat[B, F] : float16 or float32; F padded to multiple of 16.

* FieldIO:

    * input_feat[B, F]
    * sigma[B] : float32
    * rgb_feat[B, R] or rgb[B, 3] : float16/float32

* GridState:

    * bitset[L, Gpack] : uint32 or uint64
    * ema_tau : float32
    * threshold : float32
    * levels L, packed size Gpack implementation-defined; query returns bool mask.

* ArtifactMeta (python dict to be serialized):

    * version : string
    * endianness : string
    * alignment_bytes : int (>=128)
    * arch_req : string
    * precision : {weights: string, accum: string}
    * seed : int

Provide `assert_shapes_*` helpers to check SISO invariants at module boundaries.

### 2.2 `config.py`

Dataclasses holding all explicit knobs:

* DatasetConfig: root="data/nerf-synthetic", scene="lego", train_split="transforms_train.json", val_split="transforms_val.json", scale, aabb.
* ModelConfig: pe_bands, hash_levels, hash_res0, hash_per_level_scale, mlp_width, mlp_depth, activation, view_dependent.
* IntegratorConfig: step_strategy=("fixed"|"grid"), n_steps_fixed, sigma_thresh, early_stop_T.
* GridConfig: resolution, ema_tau, threshold, update_every.
* PrecisionConfig: param_dtype, compute_dtype, accum_dtype, use_amp.
* TrainConfig: batch_rays, lr, betas, weight_decay, iters, seed, deterministic.

All modules read config; no module invents its own policy.

---

## 3) Modules and responsibilities

### 3.1 `rays/`

* `cameras.py`: load intrinsics/poses from `data/nerf-synthetic/<scene>/<transforms_*.json>`, normalize scale, compute near/far.
* `rays.py`: `make_rays(H, W, cameras) -> RayBatch SoA`, optional mask/AABB clip.

### 3.2 `encoder/`

* `pe.py`: positional encoding `encode(xyz) -> feat` with configurable bands.
* `hashgrid_torch.py`: minimal dense-tensor hashgrid approximation with trilinear interpolation; multi-level concatenation; purely torch ops.

### 3.3 `field/`

* `mlp.py`: vanilla MLP with fixed depth/width; fp16 params (optional), fp32 accum; supports `forward(feat)->(sigma, rgb_feat)`.
* `heads.py`: map rgb_feat (+ optional viewdir) to rgb.

### 3.4 `integrator/`

* `marcher.py`:

    * fixed-step marcher: sample along rays, call encoder->field, apply compositor.
    * grid-step marcher: call occupancy query to skip intervals; same interface.
* `compositor.py`:

    * convert sigma to alpha per step
    * cumulative transmittance
    * rgb accumulation with early-stop on T < early_stop_T

### 3.5 `grid/`

* `occupancy.py`:

    * `query(xyz) -> mask`
    * `update(stats) -> GridState`
    * minimal EMA and thresholding; bitset in torch tensors.

### 3.6 `loss/`

* `rgb.py`: L2 RGB loss only for v1.0.

### 3.7 `opt/`

* `adam.py`: wrapper returning a `torch.optim.Adam` configured from `TrainConfig`.

### 3.8 `runtime/`

* `infer.py`: `render_batch(rays, modules, cfg) -> rgb, aux`.
* `train.py`: `train_step(batch, modules, cfg) -> metrics, states`. Uses AMP autocast and GradScaler if enabled.

### 3.9 `artifact/`

* `export_v0.py`: write `meta.json`, `topo.bin`, `params.bin`, `hashgrid.bin`, `occgrid.bin` with the minimal info required by a later CUDA runtime. Validate alignment and dtype tags.

### 3.10 Cross-cutting

* `device.py`: choose torch device, set global dtype, AMP policy.
* `rng.py`: seed torch, numpy; set torch.backends for determinism.

---

## 4) Control flow (high level)

Training step:

1. Sample a batch of rays (indices) and ground truth RGB from dataset tensors on device.
2. March along rays with fixed-step or grid-step (config).
3. Compute RGB loss.
4. Backward (AMP if enabled).
5. Optimizer step with GradScaler if AMP.
6. Optionally update occupancy EMA on schedule.

Rendering:

1. Build RayBatch SoA for target H×W on device.
2. March and compose RGB.

Artifact export:

1. Serialize meta/config, topology, parameters, hashgrid tables, and occupancy state to directory; write checksums.

---

## 5) Tests (exhaustive; pytest)

### 5.1 Conventions

* Use `pytest` markers:

    * `@pytest.mark.quick` runs < 60s on a single GPU.
    * `@pytest.mark.slow` can run several minutes.
    * `@pytest.mark.perf` requires GPU; skips if not available.
* All tests must be deterministic with `seed=1337`.
* All numeric tests use explicit tolerances; all shape tests use strict asserts.

### 5.2 Test index and purpose

1. `test_types_and_config.py`

    * Validate dataclass defaults and required fields.
    * Assert SISO helpers catch mismatched SoA lengths and misaligned feature widths.

2. `test_rng_and_device.py`

    * Set seed; generate small random tensors twice; assert identical.
    * Verify AMP policy toggles compute dtype as configured.

3. `test_rays_and_cameras.py`

    * Load `data/nerf-synthetic/lego/transforms_train.json`.
    * Build rays for a small resolution (e.g., 64×64).
    * Assert SoA lengths equal; directions normalized; near<far; device consistency.

4. `test_encoder_pe.py`

    * Encode fixed xyz; compare against a reference CPU numpy implementation; tolerance `1e-6`.
    * Property: doubling bands increases feature width as expected.

5. `test_encoder_hashgrid.py`

    * For a single level, check trilinear interpolation equals hand-computed weights on a small cube.
    * Multi-level concatenation shape check; dtype and device preserved.

6. `test_field_mlp_and_heads.py`

    * Forward a tiny batch; assert output shapes and dtypes; gradients exist for all params.
    * Activation monotonicity sanity (e.g., softplus positivity for sigma head).

7. `test_integrator_compositor_analytic.py`

    * Replace field with a deterministic stub returning constant sigma and constant rgb.
    * Closed-form: `alpha = 1 - exp(-sigma*dt)`, `T_k = prod(1 - alpha_i)`; assert compositor result equals analytic within `1e-5`.
    * Early-stop threshold respected.

8. `test_integrator_training_loop_sanity.py`

    * Tiny scene crop: sample 2k rays from lego train set; run 200 steps; assert PSNR improves by at least `+1.0 dB` compared to step 0.
    * Ensure loss decreases monotonically for a windowed average.

9. `test_grid_occupancy.py`

    * Initialize empty grid; run update with synthetic hit statistics; assert bitset toggles according to threshold.
    * With grid-step enabled, assert average steps per ray decreases vs fixed-step, and PSNR drift `<= 0.1 dB` on a small render.

10. `test_loss_rgb.py`

    * Compare against numpy L2; check dtype forwarding and reduction behavior.

11. `test_opt_adam.py`

    * One-step update on a simple quadratic; assert parameter decreases loss.
    * AMP on/off produces identical fp32 master weights within `1e-6` after unscale.

12. `test_runtime_train_step.py`

    * Assemble modules via `factory.py`; run a single `train_step`; assert grads nonzero, params updated, metrics keys present.

13. `test_artifact_export_v0.py`

    * Export artifact to a temp dir; check presence of `meta.json`, `topo.bin`, `params.bin`, `hashgrid.bin`, `occgrid.bin`.
    * Validate `meta.json` fields, alignment multiples, nonzero sizes; CRC or size sanity.

14. `test_end_to_end_render_psnr_quick.py`

    * Render 64×64 validation views for lego before and after a short train; assert PSNR >= `20 dB` quick threshold and improves by `>= 1 dB`.
    * This is a quick gate, not a full 50k-iter target.

15. `test_grad_check_small_batch.py`

    * Freeze RNG; pick 32 rays; compute finite-difference gradients for a small MLP on CPU and compare with autograd on GPU (or both CPU if needed).
    * Cosine similarity `>= 0.999`, max relative error `< 1e-3`.

16. `test_reproducibility.py`

    * Two fresh runs with the same seed and config produce identical loss curves for the first 10 steps (tolerance `1e-7` per step), identical renders bitwise or within `1e-6`.

17. `test_perf_microbench.py`

    * On a single GPU, time a render of 128×128 rays with fixed-step; record rays/s.
    * Set a conservative lower bound (e.g., `>= 1e7 rays/s` on a typical modern GPU); skip if no CUDA.

### 5.3 Test data policy

* Only `data/nerf-synthetic/` is used.
* Tests use tiny crops and reduced step counts to meet quick constraints unless marked `slow` or `perf`.

### 5.4 Running tests

* Quick suite: `pytest -q -m quick`
* Full numeric suite: `pytest -q`
* Performance: `pytest -q -m perf`

---

## 6) Implementation order (agent checklist)

1. Create `pyproject.toml` with dependencies: `torch`, `numpy`, `pytest` (dev).
2. Implement `types.py` SISO contracts and `assert_shapes_*` helpers.
3. Implement `config.py` dataclasses with defaults for lego.
4. Implement `rng.py` to fix seeds and determinism flags; implement `device.py` for AMP/dtype.
5. Implement `rays/cameras.py` loader for nerf-synthetic JSON; `rays/rays.py` to build SoA.
6. Implement `encoder/pe.py`.
7. Implement `field/mlp.py` and `field/heads.py` with fp16 params + fp32 accum option.
8. Implement `integrator/compositor.py` then `integrator/marcher.py` (fixed-step).
9. Implement `loss/rgb.py`.
10. Implement `runtime/train.py` with AMP autocast and GradScaler; `runtime/infer.py`.
11. Run quick tests for completed modules.
12. Implement `encoder/hashgrid_torch.py`, multi-level concat.
13. Implement `grid/occupancy.py`; add grid-step mode to marcher through injected query.
14. Implement `opt/adam.py` wrapper.
15. Implement `artifact/export_v0.py`.
16. Run complete test suite; adjust tolerances minimally if needed.

---

## 7) Performance guidance (PyTorch-only baseline)

* Precompute RayBatch on device; avoid Python loops; use vectorized ops only.
* Use AMP (`torch.cuda.amp.autocast`) for forward; keep accumulators in float32; scale loss with GradScaler.
* Keep feature widths and batch sizes aligned to multiples of 16 or 32.
* Avoid repeated host-device transfers; keep dataset tensors on device for training.
* Early-stop when transmittance T < threshold to save compute.

---

## 8) Artifact v0 content (for later CUDA runtime)

* `meta.json`: version, endianness, alignment, arch_req, precision, seed.
* `topo.bin`: encoder levels, bands, mlp width/depth/activation, head types.
* `params.bin`: ordered layer parameters; dtype tags; alignment padding.
* `hashgrid.bin`: levels, resolutions, offsets; per-level scaling constants.
* `occgrid.bin`: bitset snapshot; ema and threshold constants.

All binary files must be aligned to multiples of 256 bytes; sizes recorded in `meta.json`.

---

## 9) Acceptance gates

* Quick numeric correctness: all tests except `slow` and `perf` must pass on a single GPU.
* Reproducibility: deterministic tests must pass bitwise or within stated tolerances.
* Performance: microbench threshold must pass on a typical modern GPU; otherwise the test is marked `xfail` with device-specific rationale.

---

## 10) Minimal config defaults (lego)

* DatasetConfig: root `data/nerf-synthetic`, scene `lego`.
* ModelConfig: pe_bands=10, hash_levels=0 in v1.0 baseline (enable after hashgrid), mlp_width=64, mlp_depth=4, activation `softplus`, view_dependent=False.
* IntegratorConfig: step_strategy `fixed`, n_steps_fixed=96, early_stop_T=1e-3.
* GridConfig: resolution=128, ema_tau=0.1, threshold=0.01, update_every=16 (used when grid mode enabled).
* PrecisionConfig: param_dtype=float16, compute_dtype=float16, accum_dtype=float32, use_amp=True.
* TrainConfig: batch_rays=4096, lr=1e-3, betas=(0.9,0.999), weight_decay=0.0, iters=20000, seed=1337, deterministic=True.

---

## 11) File-level required public functions (names only)

* `rays.cameras.load_nerf_synthetic(scene_path) -> cameras_struct`
* `rays.rays.make_rays(H, W, cameras_struct) -> RayBatch`
* `encoder.pe.encode(xyz) -> EncodedFeat`
* `encoder.hashgrid_torch.encode(xyz, state) -> EncodedFeat`
* `field.mlp.forward(feat) -> (sigma, rgb_feat)`
* `field.heads.rgb(rgb_feat, viewdir=None) -> rgb`
* `integrator.compositor.compose(sigma_steps, rgb_steps, dt, T_threshold) -> rgb_out, T_out`
* `integrator.marcher.render_batch(rays, encoder, field, compositor, grid=None, cfg=...) -> rgb, aux`
* `grid.occupancy.query(xyz) -> mask`
* `grid.occupancy.update(stats) -> GridState`
* `loss.rgb.l2(pred, target) -> scalar`
* `opt.adam.wrap(params, cfg) -> torch.optim.Optimizer`
* `runtime.infer.render_batch(...) -> rgb, aux`
* `runtime.train.train_step(batch, modules, cfg) -> metrics, states`
* `artifact.export_v0.export(modules, cfg, out_dir) -> None`

All strings and identifiers must be ASCII.

---

## 12) Running sequence (for an agent)

1. Install: `pip install -e .`
2. Place dataset in `data/nerf-synthetic/lego/...`.
3. Run quick tests: `pytest -q -m quick`
4. Implement remaining modules; run `pytest -q`
5. Optionally run perf: `pytest -q -m perf`

---

This document is the single source of truth for `ngp-baseline-torch` v1.0. Implement strictly according to the SISO contracts, pass all tests in `tests/`, and keep the code minimal and decoupled.
