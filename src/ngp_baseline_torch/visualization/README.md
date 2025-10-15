# NGP Visualization Module

实时神经图形基元（Instant NGP）训练可视化模块，通过共享内存实现零拷贝的跨进程数据传输。

## 功能特性

- **零拷贝数据传输**: 使用 SHMX 共享内存库，避免序列化开销
- **实时可视化**: 训练过程中即时查看密度场、颜色场和训练指标
- **异步非阻塞**: 可视化不影响训练性能
- **多数据类型支持**: 点云、密度场、相机信息、训练统计等
- **跨进程通信**: Python训练进程 → C++/Vulkan可视化进程

## 安装依赖

```bash
# 安装 SHMX 共享内存库
pip install shmx

# 或从源码安装
pip install -e .
```

## 快速开始

### 1. 基础使用示例

```python
from ngp_baseline_torch.visualization import NGPDebugServer
import numpy as np

# 创建并初始化服务器
server = NGPDebugServer(
    name="ngp_debug",
    max_points=100_000,
    slots=4,
)

if server.initialize():
    # 在训练循环中发布数据
    for iteration in range(1000):
        # 生成调试数据
        positions = np.random.randn(10000, 3).astype(np.float32)
        colors = np.random.rand(10000, 3).astype(np.float32)
        densities = np.random.rand(10000, 1).astype(np.float32)
        
        # 发布帧
        server.publish_frame(
            iteration=iteration,
            positions=positions,
            colors=colors,
            densities=densities,
            loss=0.05,
            psnr=28.5,
            learning_rate=1e-3,
        )
    
    server.shutdown()
```

### 2. 与PyTorch集成

```python
import torch
from ngp_baseline_torch.visualization import (
    NGPDebugServer,
    sample_density_grid,
    filter_by_density_threshold,
)

# 初始化服务器
server = NGPDebugServer(name="ngp_training")
server.initialize()

# 在训练循环中
for iteration in range(num_iters):
    # ... 执行训练步骤 ...
    
    # 每隔N次迭代采样并发布数据
    if iteration % 10 == 0:
        with torch.no_grad():
            # 采样密度场
            positions, colors, densities = sample_density_grid(
                encoder=encoder,
                field=field,
                rgb_head=rgb_head,
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
            server.publish_frame(
                iteration=iteration,
                positions=positions,
                colors=colors,
                densities=densities,
                loss=metrics['loss'],
                psnr=metrics['psnr'],
                learning_rate=optimizer.param_groups[0]['lr'],
            )

server.shutdown()
```

### 3. 使用上下文管理器

```python
from ngp_baseline_torch.visualization import NGPDebugServer

# 自动清理资源
with NGPDebugServer(name="ngp_debug") as server:
    for iteration in range(1000):
        # ... 发布数据 ...
        server.publish_frame(...)
# 自动调用 shutdown()
```

## 完整训练示例

项目包含完整的训练示例，展示如何集成可视化：

```bash
# 运行带可视化的训练
python example_train_with_visualization.py --scene lego --iters 10000

# 禁用可视化
python example_train_with_visualization.py --scene lego --no-vis

# 自定义可视化参数
python example_train_with_visualization.py \
    --scene lego \
    --vis-interval 20 \
    --vis-max-points 200000 \
    --vis-threshold 0.05
```

## 数据流定义

### Stream ID 分配

| Stream ID | 名称 | 数据类型 | 维度 | 说明 |
|-----------|------|---------|------|------|
| 1 | frame_seq | uint64 | 1 | 帧序列号 |
| 2 | timestamp | float64 | 1 | 时间戳 |
| 3 | iteration | uint32 | 1 | 训练迭代数 |
| 100 | positions | float32 | 3 | 3D位置 (x,y,z) |
| 101 | colors | float32 | 3 | RGB颜色 |
| 102 | normals | float32 | 3 | 法向量 |
| 200 | density | float32 | 1 | 体密度值 |
| 201 | opacity | float32 | 1 | 透明度 |
| 300 | loss | float32 | 1 | 损失值 |
| 301 | psnr | float32 | 1 | PSNR指标 |
| 302 | learning_rate | float32 | 1 | 学习率 |
| 400 | camera_pos | float32 | 3 | 相机位置 |
| 401 | camera_target | float32 | 3 | 相机目标点 |
| 402 | camera_matrix | float32 | 16 | 4x4变换矩阵 |

## API 参考

### NGPDebugServer

主要的调试服务器类。

#### 构造函数

```python
NGPDebugServer(
    name: str = "ngp_debug",
    max_points: int = 500_000,
    max_rays: int = 4096,
    enable_volume: bool = False,
    volume_resolution: int = 128,
    slots: int = 4,
    reader_slots: int = 16,
)
```

**参数:**
- `name`: 共享内存区域名称
- `max_points`: 每帧最大点数
- `max_rays`: 每帧最大光线数
- `enable_volume`: 是否支持体积数据
- `volume_resolution`: 体积数据分辨率
- `slots`: 环形缓冲区槽位数
- `reader_slots`: 最大并发客户端数

#### 方法

**initialize() -> bool**

初始化共享内存服务器。

**publish_frame(...) -> bool**

发布一帧调试数据。

```python
server.publish_frame(
    iteration: int,
    positions: Optional[Tensor] = None,
    colors: Optional[Tensor] = None,
    densities: Optional[Tensor] = None,
    normals: Optional[Tensor] = None,
    loss: Optional[float] = None,
    psnr: Optional[float] = None,
    learning_rate: Optional[float] = None,
    camera_pos: Optional[ndarray] = None,
    camera_target: Optional[ndarray] = None,
    camera_matrix: Optional[ndarray] = None,
)
```

**shutdown()**

关闭服务器并释放共享内存。

### 工具函数

#### sample_density_grid()

从训练好的网络中采样密度和颜色场。

```python
positions, colors, densities = sample_density_grid(
    encoder=encoder,
    field=field,
    rgb_head=rgb_head,
    bbox_min=np.array([-1, -1, -1]),
    bbox_max=np.array([1, 1, 1]),
    num_samples=100_000,
    device=device,
)
```

#### filter_by_density_threshold()

根据密度阈值过滤点云。

```python
positions, colors, densities = filter_by_density_threshold(
    positions, colors, densities,
    threshold=0.01,
    max_points=50_000,
)
```

#### extract_training_metrics()

提取训练指标。

```python
metrics = extract_training_metrics(
    metrics=training_metrics,
    optimizer=optimizer,
)
```

## 性能考虑

### 最佳实践

1. **更新频率**: 不要每次迭代都更新可视化，推荐每10-50次迭代更新一次
2. **点数限制**: 限制发送的点数（建议10万-50万），使用密度阈值过滤
3. **异步传输**: GPU→CPU传输会有开销，但通常很小（<1ms）
4. **批处理**: 使用较大的batch_size采样以提高效率

### 性能示例

```python
# 推荐配置
server = NGPDebugServer(
    max_points=100_000,  # 限制点数
    slots=4,             # 4个缓冲槽足够
)

# 每20次迭代更新一次
if iteration % 20 == 0:
    with torch.no_grad():
        # 采样并过滤
        positions, colors, densities = sample_density_grid(...)
        positions, colors, densities = filter_by_density_threshold(
            ..., threshold=0.01, max_points=50_000
        )
        server.publish_frame(...)
```

## 客户端集成

虽然本模块只实现服务器端，但客户端可以通过SHMX C++ API连接：

```cpp
#include "shmx_client.h"

// 连接到共享内存
shmx::Client client;
client.connect("ngp_debug");

// 轮询最新帧
while (running) {
    auto frame_view = client.poll_latest_frame();
    if (frame_view.has_value()) {
        // 解码并渲染数据
        shmx::DecodedFrame decoded;
        shmx::Client::decode(frame_view.value(), decoded);
        
        // 访问流数据（零拷贝）
        auto& positions = decoded.streams[100];  // Stream ID 100
        auto& colors = decoded.streams[101];
        // ... 渲染 ...
    }
}
```

## 调试与监控

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

server = NGPDebugServer(name="ngp_debug")
server.initialize()
```

### 检查服务器状态

```python
# 获取帧计数
print(f"Published frames: {server.frame_count}")

# 检查是否启用
print(f"Enabled: {server.is_enabled()}")

# 临时禁用（不影响连接）
server.set_enabled(False)
# ... 训练一段时间 ...
server.set_enabled(True)
```

### 接收控制消息

```python
# 客户端可以发送控制消息给服务器
messages = server.poll_control_messages()
for msg in messages:
    print(f"Message from reader {msg['reader_id']}: {msg}")
```

## 示例脚本

项目包含多个示例脚本：

- `example_debug_server.py`: 独立的服务器使用示例
- `example_train_with_visualization.py`: 完整的训练+可视化示例

运行示例：

```bash
# 基础示例
python example_debug_server.py

# PyTorch集成示例
python example_debug_server.py --torch

# 上下文管理器示例
python example_debug_server.py --context
```

## 故障排除

### shmx未安装

```
Error: shmx library not available
```

**解决方案**: 安装shmx库
```bash
pip install shmx
```

### 初始化失败

```
Failed to initialize server!
```

**可能原因**:
1. 同名共享内存已存在（使用不同的name或重启系统）
2. 权限不足
3. 共享内存大小超限

**解决方案**:
```python
# 使用唯一名称
import time
server = NGPDebugServer(name=f"ngp_{int(time.time())}")
```

### GPU内存不足

采样大量点时可能导致GPU内存不足。

**解决方案**:
```python
# 减少采样点数
positions, colors, densities = sample_density_grid(
    num_samples=50_000,  # 减少数量
    batch_size=4096,     # 减少批大小
)
```

## 许可证

本模块遵循项目主许可证。

## 贡献

欢迎提交问题和拉取请求！

