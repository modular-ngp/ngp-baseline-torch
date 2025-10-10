# 测试系统说明

## 📋 新测试架构

测试系统已完全重写，删除了冗余测试，保留关键验证，并添加了严格的收敛性测试。

### 测试文件结构

```
tests/
├── conftest.py                    # Pytest配置
├── test_1_core_components.py      # 核心组件测试
├── test_2_integrator.py           # 积分器测试
├── test_3_training_components.py  # 训练组件测试
└── test_4_e2e_convergence.py      # ⚠️ 端到端收敛测试（关键）
```

---

## ✅ Test 1: 核心组件 (test_1_core_components.py)

### TestHashGridEncoder
- ✓ **初始化范围检查** (CRITICAL)
  - 验证所有hash table参数在 `[-1e-4, 1e-4]` 范围内
  - 这是最关键的修复之一
- ✓ **前向传播形状**
  - 验证输出维度正确且已padding到16的倍数
- ✓ **梯度流动**
  - 确保梯度能正确传播到hash table

### TestMLP
- ✓ **截断指数激活** (CRITICAL)
  - 验证使用 `trunc_exp` 而非 `softplus`
  - 检查密度输出非负且有界
- ✓ **Sigma偏置初始化** (CRITICAL)
  - 验证sigma head的bias为 `-1.5`
  - 确保初始密度接近0（稀疏初始化）
- ✓ **输出形状**

### TestRGBHead
- ✓ **视角依赖模式**
  - 验证view-dependent渲染工作正常
- ✓ **输出范围**
  - RGB值在 `[0, 1]` 范围内

---

## ✅ Test 2: 积分器 (test_2_integrator.py)

### TestRayMarcher
- ✓ **采样抖动** (CRITICAL)
  - 验证训练模式下启用jittering
  - 验证eval模式下禁用jittering
  - 这是减少aliasing的关键
- ✓ **采样数量**
  - 确保生成正确数量的采样点

### TestVolumeCompositing
- ✓ **基础合成**
  - alpha compositing正确性
- ✓ **Early stopping**
  - 高密度区域能提前终止
- ✓ **白色背景**
  - NeRF-Synthetic场景的白背景处理

---

## ✅ Test 3: 训练组件 (test_3_training_components.py)

### TestOptimizer
- ✓ **分离学习率**
  - Encoder: 1e-2 (高学习率)
  - MLP: 1e-3 (较低学习率)
- ✓ **参数更新**
  - 优化器能正确更新参数

### TestLoss
- ✓ **RGB L2损失**
- ✓ **MSE到PSNR转换**

### TestGradients
- ✓ **端到端梯度流**
  - 从loss到encoder的完整梯度传播

---

## ⚠️ Test 4: 端到端收敛 (test_4_e2e_convergence.py) - **关键测试**

这是整个测试系统中**最重要**的测试，验证模型是否能达到Instant-NGP的理论性能。

### TestQuickOverfit
```python
test_single_image_overfit()
```
- **目标**: 500次迭代，单张图片过拟合
- **期望PSNR**: >25 dB
- **用途**: 快速验证模型基本功能

### TestFastConvergence
```python
test_1000_iterations_convergence()
```
- **目标**: 1000次迭代，完整训练集
- **期望PSNR**: >20 dB
- **用途**: 验证快速收敛能力

### TestFullConvergence ⭐ **最关键**
```python
test_5000_iterations_target_psnr()
```
- **场景**: Lego (NeRF-Synthetic)
- **迭代次数**: 5000
- **期望PSNR**: **≥28.0 dB**
- **理论依据**: 
  - Instant-NGP论文报告Lego场景最终能达到32+ dB
  - 5000次迭代应该达到28+ dB
  - 20000次迭代应该达到30+ dB

#### 里程碑检查点：
| 迭代 | 期望PSNR | 容差 |
|------|---------|------|
| 100  | ~15 dB  | ±2   |
| 500  | ~20 dB  | ±2   |
| 1000 | ~22 dB  | ±2   |
| 2000 | ~25 dB  | ±2   |
| 5000 | **≥28 dB** | 严格 |

**如果此测试失败（<28 dB），说明存在根本性实现问题。**

---

## 🚀 运行测试

### 方法1: 使用测试运行脚本（推荐）

```bash
# 快速测试（不包含收敛测试）
python run_tests.py

# 快速收敛检查（500迭代，单图过拟合）
python run_tests.py --quick

# 完整收敛测试（5000迭代，关键测试）⭐
python run_tests.py --full

# 运行所有测试
python run_tests.py --all
```

### 方法2: 直接使用pytest

```bash
# 运行所有快速测试
pytest tests/ -v -m "not slow"

# 运行特定测试
pytest tests/test_1_core_components.py -v

# 运行关键收敛测试
pytest tests/test_4_e2e_convergence.py::TestFullConvergence -v -s

# 运行所有测试（包括慢速）
pytest tests/ -v -s
```

---

## 📊 预期输出

### 成功的收敛测试输出示例：

```
======================================================================
CRITICAL CONVERGENCE TEST - 5000 Iterations on Lego Scene
Target: 28+ dB (Instant-NGP baseline)
Device: cuda
======================================================================

Training set: 100 images, 640,000 rays

  Iter  100: PSNR=14.23 dB, Avg(100)=14.23 dB, LR=0.010000
  Iter  500: PSNR=19.87 dB, Avg(100)=19.45 dB, LR=0.010000
  Iter 1000: PSNR=22.34 dB, Avg(100)=22.01 dB, LR=0.010000
  Iter 2000: PSNR=25.67 dB, Avg(100)=25.42 dB, LR=0.007499
  Iter 5000: PSNR=28.89 dB, Avg(100)=28.45 dB, LR=0.001000

======================================================================
RESULTS:
  Final Avg PSNR (last 100): 28.45 dB
  Max PSNR (last 500):       28.89 dB
  Target:                     28.00 dB
======================================================================

✓ CONVERGENCE TEST PASSED: 28.45 dB ≥ 28.0 dB
```

---

## ⚠️ 故障排查

### 如果Test 4失败（PSNR < 28 dB）

检查以下关键点：

1. **Hash Grid初始化**
   ```python
   # 应该是 uniform(-1e-4, 1e-4)
   pytest tests/test_1_core_components.py::TestHashGridEncoder::test_initialization_range -v
   ```

2. **密度激活函数**
   ```python
   # 应该是 trunc_exp
   pytest tests/test_1_core_components.py::TestMLP::test_density_activation_trunc_exp -v
   ```

3. **采样抖动**
   ```python
   # 训练时应该启用
   pytest tests/test_2_integrator.py::TestRayMarcher::test_sample_jittering -v
   ```

4. **学习率设置**
   ```python
   # Encoder: 1e-2, MLP: 1e-3
   pytest tests/test_3_training_components.py::TestOptimizer::test_separate_learning_rates -v
   ```

### 常见失败原因

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| PSNR停在15-18 | Hash grid初始化错误 | 检查uniform分布 |
| PSNR停在20-23 | 密度激活函数错误 | 使用trunc_exp |
| 训练不稳定 | 缺少采样抖动 | 启用perturb=True |
| 收敛非常慢 | 学习率过低 | Encoder用1e-2 |

---

## 📈 性能基准

在NVIDIA RTX 3090上的预期性能：

| 测试 | 时间 | 内存 |
|------|------|------|
| Test 1-3 (快速) | ~30秒 | ~2GB |
| 单图过拟合 (500) | ~2分钟 | ~4GB |
| 快速收敛 (1000) | ~5分钟 | ~5GB |
| **完整收敛 (5000)** | **~20分钟** | **~6GB** |

---

## 🎯 质量保证

### 必须通过的测试

1. ✅ Test 1: 所有核心组件测试
2. ✅ Test 2: 所有积分器测试
3. ✅ Test 3: 所有训练组件测试
4. ⚠️ **Test 4: test_5000_iterations_target_psnr** ← **关键质量门槛**

**只有当Test 4通过（≥28 dB）时，才能认为实现是正确的。**

---

## 🔄 持续集成建议

```yaml
# .github/workflows/test.yml 示例
- name: Fast Tests
  run: python run_tests.py

- name: Quick Convergence
  run: python run_tests.py --quick

- name: Full Convergence (Nightly)
  run: python run_tests.py --full
  if: github.event_name == 'schedule'
```

---

## 📝 添加新测试

如果需要添加新测试：

1. 将测试添加到相应的test_N文件
2. 对于慢速测试，添加 `@pytest.mark.slow` 装饰器
3. 确保测试有明确的断言和错误信息
4. 更新此README

---

## ✨ 总结

新测试系统的核心理念：

1. **删除冗余** - 从18个测试文件减少到4个
2. **关注关键** - 测试最重要的修复点
3. **严格验证** - Test 4提供严格的性能门槛（28+ dB）
4. **快速反馈** - 分层测试（快速→中等→慢速）

**Test 4 (5000迭代→28+ dB) 是整个项目质量的最终保证。**

