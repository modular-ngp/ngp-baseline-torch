# NGP Visualization Implementation - 实现说明

## 概述

本项目已成功实现基于 SHMX 共享内存的 Instant NGP 训练可视化 Python 服务器端。

## 实现的功能模块

### 1. 核心模块结构

```
src/ngp_baseline_torch/visualization/
├── __init__.py           # 模块导出
├── debug_server.py       # NGPDebugServer 主类
├── debug_extractor.py    # 调试数据提取工具
└── README.md            # 详细文档
```

### 2. NGPDebugServer 类

**位置**: `src/ngp_baseline_torch/visualization/debug_server.py`

**主要功能**:
- 创建和管理 SHMX 共享内存服务器
- 定义数据流规范（Stream Specifications）
- 零拷贝发布训练调试数据
- 支持 PyTorch Tensor 和 NumPy Array
- 自动处理 GPU→CPU 传输

**核心方法**:
```python
# 初始化服务器
server.initialize() -> bool

# 发布一帧数据
server.publish_frame(
    iteration: int,
    positions: Tensor/Array,
    colors: Tensor/Array,
    densities: Tensor/Array,
    loss: float,
    psnr: float,
    learning_rate: float,
    ...
) -> bool

# 关闭服务器
server.shutdown()
```

**支持的数据流** (Stream IDs):
- **元数据** (1-3): 帧序列号、时间戳、迭代数
- **几何数据** (100-102): 位置、颜色、法向量
- **场数据** (200-202): 密度、透明度、特征
- **训练统计** (300-302): 损失、PSNR、学习率
- **相机数据** (400-402): 位置、目标点、变换矩阵

### 3. 调试数据提取工具

**位置**: `src/ngp_baseline_torch/visualization/debug_extractor.py`

**工具函数**:

#### `sample_density_grid()`
从训练的 NGP 模型中采样密度和颜色场：
```python
positions, colors, densities = sample_density_grid(
    encoder=encoder,
    field=field,
    rgb_head=rgb_head,
    bbox_min=[-1, -1, -1],
    bbox_max=[1, 1, 1],
    num_samples=100_000,
    device=device,
)
```

#### `filter_by_density_threshold()`
根据密度阈值过滤点云，减少可视化数据量：
```python
positions, colors, densities = filter_by_density_threshold(
    positions, colors, densities,
    threshold=0.01,
    max_points=50_000,
)
```

#### `extract_training_metrics()`
提取训练指标用于可视化：
```python
metrics = extract_training_metrics(
    metrics=training_metrics,
    optimizer=optimizer,
)
```

#### `sample_rays_along_camera()`
沿相机光线采样点用于调试：
```python
ray_origins, ray_directions, sample_positions = sample_rays_along_camera(
    rays=rays,
    num_rays=1024,
    num_samples_per_ray=64,
)
```

### 4. 示例脚本

#### `example_debug_server.py`
独立的调试服务器示例，演示基本用法：
```bash
# 基础示例
python example_debug_server.py

# PyTorch 集成
python example_debug_server.py --torch

# 上下文管理器
python example_debug_server.py --context
```

#### `example_train_with_visualization.py`
完整的训练+可视化集成示例：
```bash
# 运行训练并启用可视化
python example_train_with_visualization.py --scene lego --iters 10000

# 自定义可视化参数
python example_train_with_visualization.py \
    --scene lego \
    --vis-interval 20 \
    --vis-max-points 200000 \
    --vis-threshold 0.05

# 禁用可视化
python example_train_with_visualization.py --scene lego --no-vis
```

#### `test_visualization.py`
测试脚本，验证模块功能：
```bash
python test_visualization.py
```

## 使用流程

### 步骤 1: 安装依赖

```bash
# 安装 SHMX 共享内存库（已上传到 PyPI）
pip install shmx
```

### 步骤 2: 在训练代码中集成

```python
from ngp_baseline_torch.visualization import (
    NGPDebugServer,
    sample_density_grid,
    filter_by_density_threshold,
)

# 初始化服务器
debug_server = NGPDebugServer(
    name="ngp_training",
    max_points=100_000,
)

if debug_server.initialize():
    print("Visualization server ready!")

# 训练循环
for iteration in range(num_iters):
    # 执行训练步骤
    metrics = trainer.step(ray_batch, target_batch)
    
    # 每隔 N 次迭代更新可视化
    if iteration % 10 == 0:
        with torch.no_grad():
            # 采样密度场
            positions, colors, densities = sample_density_grid(
                encoder, field, rgb_head,
                num_samples=100_000,
                device=device,
            )
            
            # 过滤低密度点
            positions, colors, densities = filter_by_density_threshold(
                positions, colors, densities,
                threshold=0.01,
                max_points=50_000,
            )
            
            # 发布到共享内存
            debug_server.publish_frame(
                iteration=iteration,
                positions=positions,
                colors=colors,
                densities=densities,
                loss=metrics['loss'],
                psnr=metrics['psnr'],
                learning_rate=optimizer.param_groups[0]['lr'],
            )

# 训练结束后清理
debug_server.shutdown()
```

### 步骤 3: 在另一个进程中运行可视化客户端

客户端部分您将在另一个项目中实现（C++/Vulkan），使用 SHMX C++ API：

```cpp
#include "shmx_client.h"

// 连接到共享内存
shmx::Client client;
if (client.connect("ngp_training")) {
    // 轮询最新帧
    while (running) {
        auto frame = client.poll_latest_frame();
        if (frame.has_value()) {
            // 解码数据
            shmx::DecodedFrame decoded;
            client.decode(frame.value(), decoded);
            
            // 访问位置数据（Stream ID 100）
            auto& positions = decoded.streams[100];
            float* pos_data = (float*)positions.ptr;
            size_t num_points = positions.elem_count;
            
            // 渲染点云...
        }
    }
}
```

## 技术特性

### 零拷贝传输

- 使用 `numpy.tobytes()` 直接获取内存视图
- 避免序列化/反序列化开销
- PyTorch Tensor 自动转换为连续内存

### 异步非阻塞

- `publish_frame()` 立即返回，不等待客户端
- 使用环形缓冲区，自动处理帧丢弃
- 训练性能不受可视化影响

### 自动资源管理

- 支持上下文管理器（`with` 语句）
- 自动清理共享内存
- 异常安全

### 类型灵活性

- 自动处理 PyTorch Tensor 和 NumPy Array
- 自动处理 CPU/CUDA 设备
- 自动转换数据类型（float32）

## 性能优化建议

### 1. 合理的更新频率

```python
# 推荐：每 10-50 次迭代更新一次
if iteration % 20 == 0:
    debug_server.publish_frame(...)
```

### 2. 限制点数

```python
# 使用密度阈值和最大点数限制
positions, colors, densities = filter_by_density_threshold(
    positions, colors, densities,
    threshold=0.01,      # 过滤低密度点
    max_points=50_000,   # 限制总数
)
```

### 3. 批处理采样

```python
# 使用较大的批大小提高采样效率
positions, colors, densities = sample_density_grid(
    ...,
    batch_size=8192,  # 增大批大小
)
```

### 4. 条件启用

```python
# 可以随时启用/禁用，不影响训练
debug_server.set_enabled(False)  # 暂时禁用
# ... 快速训练 ...
debug_server.set_enabled(True)   # 重新启用
```

## 数据流规范

完整的数据流定义请参考 `src/ngp_baseline_torch/visualization/README.md`。

关键点：
- 所有几何数据使用 float32 类型
- 位置/颜色/法向量为 [N, 3] 形状
- 密度为 [N, 1] 或 [N,] 形状
- 标量数据（loss, psnr 等）为单个值

## 错误处理

模块包含完善的错误处理：

```python
try:
    success = debug_server.publish_frame(...)
    if not success:
        print("Failed to publish frame")
except Exception as e:
    print(f"Error: {e}")
```

如果 SHMX 未安装，模块会优雅地降级：
```python
if not SHMX_AVAILABLE:
    print("Warning: shmx not available")
    # 继续训练，不崩溃
```

## 与现有代码集成

本实现完全独立，不修改现有训练代码：

- ✓ 不依赖现有模块的内部实现
- ✓ 可选集成，不影响正常训练
- ✓ 零性能开销（未启用时）
- ✓ 向后兼容

## 测试和验证

运行测试脚本验证实现：

```bash
# 测试模块导入和基本功能
python test_visualization.py

# 测试独立服务器
python example_debug_server.py

# 测试完整训练集成
python example_train_with_visualization.py --scene lego --iters 100
```

## 下一步

您的可视化客户端（另一个项目）需要：

1. 使用 SHMX C++ API 连接到共享内存
2. 轮询最新帧数据
3. 解码 Stream 数据
4. 使用 Vulkan 渲染点云
5. 显示训练统计（ImGui）

客户端示例请参考设计文档 `INSTANT_NGP_VISUALIZATION_PLAN.md` 的 C++ 部分。

## 总结

✅ **已完成**:
- NGPDebugServer 核心实现
- 数据提取和过滤工具
- PyTorch/NumPy 集成
- 完整示例和文档
- 测试脚本

📝 **使用方法**:
1. `pip install shmx`
2. 在训练代码中导入并初始化服务器
3. 定期发布调试数据
4. 在另一个进程中运行可视化客户端

🎯 **核心优势**:
- 零拷贝，高性能
- 异步非阻塞
- 易于集成
- 完全独立

现在您可以开始在另一个项目中实现 C++/Vulkan 可视化客户端了！

