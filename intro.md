# INTRODUCTION


## 项目概述

DeepSpeed 是一个深度学习优化软件套件，专注于大规模模型的训练和推理。核心创新包括 ZeRO（Zero Redundancy Optimizer）内存优化、3D 并行（数据/张量/流水线并行）、混合专家（MoE）系统以及高性能推理引擎。

## 常用开发命令

### 环境设置与构建
```bash
# 创建虚拟环境并安装开发依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/requirements-dev.txt

# 可编辑安装（JIT 编译扩展操作）
pip install -e .

# 预编译所有 CUDA/C++ 扩展操作（加速后续启动）
DS_BUILD_OPS=1 pip install -e .

# 检查环境和可用扩展操作
ds_report
```

### 测试
```bash
# 运行所有单元测试（推荐使用 --forked 以避免 GPU 状态污染）
make test
# 或者
pytest --forked tests/unit/

# 运行特定测试模式
pytest --forked -k "test_pattern" tests/unit/

# 运行单个测试文件
pytest --forked tests/unit/runtime/test_engine.py

# 模型收敛测试
cd tests/model/
pytest run_sanity_check.py

# 性能基准测试
pytest tests/perf/
```

### 代码格式化与检查
```bash
# 安装 pre-commit hooks（仅需一次）
pre-commit install

# 手动运行格式化（检查相对于 master 的改动）
make format

# 运行所有文件的格式化
pre-commit run --all-files

# 单独运行 yapf（Python 格式化）
yapf -i -r deepspeed/

# 运行 flake8 检查
flake8 deepspeed/
```

### Git 提交规范
```bash
# 所有提交必须签署 DCO（Developer Certificate of Origin）
git commit -s -m "component: concise summary"

# 提交信息格式：<area>: <imperative mood description>
# 例如: "runtime: optimize zero stage 3 parameter gathering"
```

## 代码架构

### 核心模块组织

**主 API 层 (`deepspeed/`)**:
- `runtime/`: 核心运行时引擎，包含 DeepSpeedEngine（主训练引擎）、ZeRO 优化器、配置系统
- `runtime/zero/`: ZeRO 内存优化的各个阶段实现（stage_1, stage_2, stage_3, zero_infinity）
- `runtime/pipe/`: 流水线并行引擎和调度器
- `runtime/hybrid_engine/`: 混合引擎，结合训练和推理功能
- `inference/`: 推理引擎和模型注入系统
- `ops/`: Python 包装的自定义操作（Adam、Transformer、稀疏注意力等）
- `moe/`: 混合专家（Mixture of Experts）系统实现
- `compression/`: 模型压缩技术（量化、剪枝）
- `module_inject/`: 动态模块替换，用于优化 HuggingFace 等框架的模型
- `launcher/`: 分布式训练启动器（多节点/多 GPU）
- `comm/`: 通信抽象层（支持 NCCL、MPI、Gloo）
- `checkpoint/`: 通用检查点系统（Universal Checkpointing）
- `autotuning/`: 自动超参数调优系统

**CUDA/C++ 内核层 (`csrc/`)**:
- `adam/`: Adam 优化器的融合 CUDA 内核
- `transformer/`: Transformer 层的高性能内核（推理和训练）
- `aio/`: 异步 I/O 操作（用于 NVMe 卸载）
- `quantization/`: 量化内核（ZeroQuant, FP6 等）
- `spatial/`: 空间并行内核
- `deepspeed4science/`: 科学计算专用内核
- 各加速器特定目录: `cpu/`, `xpu/`（Intel GPU）

**构建系统 (`op_builder/`)**:
- 每个自定义操作都有对应的构建器类（继承自 `OpBuilder`）
- 支持 JIT（即时编译）和 AOT（预编译）两种模式
- 加速器抽象：通过 `accelerator/` 目录支持 NVIDIA GPU、AMD ROCm、Intel Gaudi/XPU、Huawei NPU 等

### 关键设计模式

**DeepSpeedEngine 初始化流程**:
1. 解析 DeepSpeed 配置文件（JSON 或 Python 字典）
2. 根据配置创建优化器（可能包装为 ZeRO 优化器）
3. 设置数据并行、模型并行、流水线并行
4. 注册前向/反向钩子以实现梯度累积、损失缩放等
5. 返回引擎对象供用户调用 `.train()`, `.step()`, `.backward()` 等方法

**ZeRO 优化器分层**:
- Stage 1: 优化器状态分片
- Stage 2: 梯度分片
- Stage 3: 参数分片（最激进，支持万亿参数模型）
- ZeRO-Infinity: 利用 NVMe 和 CPU 内存进行参数/优化器状态卸载
- ZeRO++: 通信优化（量化、层次化通信）

**模块注入机制**:
- 动态替换 PyTorch 模块（如 `nn.Linear`、`nn.LayerNorm`）为优化版本
- 支持自动张量并行（AutoTP）：自动将模型转换为张量并行
- 主要用于推理加速和 HuggingFace 模型优化

**3D 并行协同**:
- 数据并行（DP）：跨数据副本梯度同步
- 张量并行（TP）：将层内参数分割到多个设备
- 流水线并行（PP）：将层间切分，形成流水线
- 三者可组合使用，通过 `runtime/pipe/topology.py` 管理拓扑

## 测试与调试

**单元测试组织** (`tests/unit/`):
- 按功能区域划分子目录：`runtime/`, `ops/`, `moe/`, `checkpoint/`, `compression/`
- 使用 `pytest --forked` 以避免 CUDA 上下文在测试间泄漏
- 许多测试需要多 GPU（通过 `@pytest.mark.gpu_count(N)` 标记）
- 使用 `dist_init_required` fixture 初始化分布式环境

**模型测试** (`tests/model/`):
- 端到端收敛测试，验证训练正确性
- 包含不同 ZeRO 阶段、混合精度、流水线并行配置的测试

**常见调试技巧**:
```bash
# 启用详细日志
export DEEPSPEED_LOG_LEVEL=DEBUG

# 禁用 JIT 编译以获得更好的错误堆栈
DS_BUILD_OPS=1 pip install -e .

# 单 GPU 调试（避免分布式复杂性）
python -m deepspeed.launcher.launch --num_gpus=1 train_script.py

# 使用 PyTorch 分布式启动器
torchrun --nproc_per_node=2 train_script.py --deepspeed ds_config.json
```

## 配置系统

DeepSpeed 使用 JSON 配置文件控制所有功能。关键配置项：

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 3e-5}
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {"warmup_min_lr": 0, "warmup_max_lr": 3e-5}
  },
  "zero_optimization": {
    "stage": 3,
    "offload_param": {"device": "cpu"},
    "overlap_comm": true
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  }
}
```

配置类位于 `runtime/config.py`（`DeepSpeedConfig`），负责验证和提供访问接口。

## 扩展操作构建

**添加新的 CUDA 操作**:
1. 在 `csrc/<op_name>/` 创建 `.cpp` 和 `.cu` 文件
2. 在 `op_builder/<op_name>.py` 创建构建器类
3. 在 `op_builder/all_ops.py` 注册新操作
4. 在 `deepspeed/ops/<op_name>/` 创建 Python 包装
5. 添加 `tests/unit/ops/` 测试
6. 使用 `DS_BUILD_<OP_NAME>=1` 环境变量控制是否构建

## 性能优化考虑

- **通信-计算重叠**: ZeRO 和流水线并行大量使用后台通信
- **融合内核**: 多个操作融合为单个 CUDA 内核（如 Adam + copy）
- **量化**: 训练时通信量化（1-bit Adam/LAMB）、推理时权重量化（ZeroQuant）
- **卸载策略**: 参数和优化器状态可卸载到 CPU 内存或 NVMe（ZeRO-Infinity）
- **编译优化**: DeepSpeed-Compile 通过 PyTorch 2.x 编译器优化整个图

## 多加速器支持

DeepSpeed 通过 `accelerator/` 抽象支持多种硬件：
- NVIDIA GPU（主要支持）
- AMD ROCm（通过 HIPify 自动转换 CUDA 代码）
- Intel Gaudi/XPU
- Huawei Ascend NPU

加速器 API 提供统一接口（`get_accelerator()`），封装设备管理、内存操作、通信原语。

## 文档与学习资源

- 官方文档: https://www.deepspeed.ai/
- 教程: https://www.deepspeed.ai/tutorials/
- API 文档: https://deepspeed.readthedocs.io/
- 博客: `blogs/` 目录包含深度技术文章
- 示例: https://github.com/deepspeedai/DeepSpeedExamples（外部仓库）

## 发版与维护

- 版本号在 `version.txt`，构建字符串由 `setup.py` 管理
- 使用 `DS_BUILD_STRING` 环境变量控制发版后缀（如 `.dev20250101`）
- CI 流程通过 `.github/workflows/` 管理，覆盖多种加速器和集成测试
- 遵循三步新功能贡献流程：提案讨论 → 实现验证 → 发布维护（见 CONTRIBUTING.md）
