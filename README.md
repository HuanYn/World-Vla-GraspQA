# World-Vla-GraspQA

基于世界模型的机器人抓取问答与闭环操作系统。

本项目目标是构建一个可逐步扩展的多模态机器人操作研究原型。当前阶段先不直接接入真实机器人、VLM 或复杂仿真，而是先搭建一个稳定、可测试、可复现的工程骨架。

当前版本已经实现了一个最小可运行的 dummy pipeline：

```text
配置文件
  ↓
Dummy 感知模块
  ↓
Dummy 抓取问答模块
  ↓
候选动作生成
  ↓
Dummy 世界模型预测
  ↓
规划器选择动作
  ↓
保存时间戳实验结果
  ↓
汇总实验结果
```

## 当前阶段能力

当前项目已经支持：

- 可运行的 dummy 端到端 pipeline。
- 使用 YAML 配置文件驱动实验。
- 支持多个 dummy 任务配置。
- 每次实验自动创建带时间戳的输出目录。
- 每次实验保存 `config.yaml` 和 `result.json`。
- 支持批量运行多个配置文件。
- 支持将所有实验结果汇总成 CSV。
- 已为核心模块和工具函数添加单元测试。
- 使用 `ruff` 和 `black` 进行代码质量检查。

## 项目结构

```text
World-Vla-GraspQA/
├── configs/
│   ├── dummy_pipeline.yaml
│   └── dummy_pipeline_banana.yaml
├── scripts/
│   ├── run_dummy_pipeline.py
│   ├── run_all_dummy_configs.py
│   └── summarize_dummy_results.py
├── src/
│   └── world_vla_graspqa/
│       ├── action/
│       ├── critic/
│       ├── graspqa/
│       ├── memory/
│       ├── perception/
│       ├── planner/
│       ├── simulation/
│       ├── utils/
│       └── world_model/
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 环境配置

创建 Conda 环境：

```bash
conda create -n wvg python=3.10 -y
conda activate wvg
```

安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

这里使用 `pip install -e .` 是为了把当前项目以 editable 模式安装到环境中。这样修改 `src/` 下的源码后，不需要重新安装项目，改动会立即生效。

## 运行单个 dummy pipeline

运行默认 red cube 任务：

```bash
python scripts/run_dummy_pipeline.py \
  --config configs/dummy_pipeline.yaml \
  --run-name cube_test
```

运行 banana 任务：

```bash
python scripts/run_dummy_pipeline.py \
  --config configs/dummy_pipeline_banana.yaml \
  --run-name banana_test
```

每次运行都会自动创建一个实验输出目录：

```text
outputs/dummy_pipeline/YYYYMMDD_HHMMSS_runname/
├── config.yaml
└── result.json
```

其中：

- `config.yaml` 保存本次实验使用的配置。
- `result.json` 保存本次实验运行后的结果。

这样做的目的是保证实验可复现。之后回看某一次结果时，可以同时知道它是用哪个配置跑出来的。

## 批量运行 dummy 配置

运行所有预定义 dummy 配置：

```bash
python scripts/run_all_dummy_configs.py
```

当前会批量运行：

```text
configs/dummy_pipeline.yaml
configs/dummy_pipeline_banana.yaml
```

批量运行后，每个配置都会在 `outputs/dummy_pipeline/` 下生成独立的时间戳目录。

## 汇总实验结果

将所有 dummy 实验结果汇总为 CSV：

```bash
python scripts/summarize_dummy_results.py
```

汇总文件会保存到：

```text
outputs/dummy_pipeline_summary.csv
```

CSV 中包含的主要字段包括：

```text
run_dir, run_name, target_object, best_action, gripper_pose, predicted_success, config_path
```

这样可以快速比较不同实验配置的输出结果。

## 运行测试和代码检查

运行单元测试：

```bash
pytest
```

运行代码质量检查：

```bash
ruff check .
black --check .
```

当前预期结果：

```text
13 passed
All checks passed
All done
```

## 当前 dummy pipeline 逻辑

当前版本使用手写规则实现完整流程：

- `DummyPerception`：从 YAML 配置文件中读取场景物体。
- `DummyGraspQA`：根据任务指令选择应该抓取的物体。
- `DummyActionGenerator`：为目标物体生成三个候选抓取动作。
- `DummyWorldModel`：为不同候选动作分配固定成功率。
- `DummyPlanner`：选择预测成功率最高的动作。

当前 dummy pipeline 的目的不是追求模型复杂度，而是先建立稳定的工程骨架。后续每个 dummy 模块都可以逐步替换为真实模块：

```text
DummyPerception       → 基于 VLM 的感知模块
DummyGraspQA          → 基于 VLM 的抓取问答模块
DummyActionGenerator  → 真实抓取候选生成器
DummyWorldModel       → 可学习的世界模型
DummyPlanner          → 闭环规划器
```

## 当前实验示例

默认 red cube 配置中，任务指令是：

```text
Pick up the red cube and place it into the blue bowl.
```

系统会选择：

```text
target_object = red cube
best_action = grasp(red cube)
gripper_pose = top_down
predicted_success = 0.85
```

banana 配置中，任务指令是：

```text
Pick up the yellow banana and place it into the blue bowl.
```

系统会选择：

```text
target_object = yellow banana
best_action = grasp(yellow banana)
gripper_pose = top_down
predicted_success = 0.85
```

这说明当前 pipeline 已经支持通过不同配置文件驱动不同任务。

## 主要文件说明

### `configs/dummy_pipeline.yaml`

默认 red cube 任务配置。

它定义了：

- 当前任务指令。
- 场景中的物体列表。
- 每个物体的颜色、形状和是否可抓取。
- GraspQA 问题。
- Planner 策略。

这个配置会让系统选择 `red cube` 作为目标物体。

### `configs/dummy_pipeline_banana.yaml`

banana 任务配置。

它和默认配置的主要区别是任务指令变成了：

```text
Pick up the yellow banana and place it into the blue bowl.
```

这个配置会让系统选择 `yellow banana` 作为目标物体。

### `scripts/run_dummy_pipeline.py`

单次实验入口脚本。

它负责：

- 读取指定 YAML 配置。
- 调用 perception、graspqa、action、world_model、planner 模块。
- 生成候选动作。
- 使用 dummy world model 为动作打分。
- 选择最优动作。
- 创建时间戳输出目录。
- 保存 `config.yaml` 和 `result.json`。

可以通过命令行参数指定配置文件和运行名：

```bash
python scripts/run_dummy_pipeline.py \
  --config configs/dummy_pipeline.yaml \
  --run-name cube_test
```

### `scripts/run_all_dummy_configs.py`

批量运行脚本。

它会自动运行多个预定义配置，例如：

- `configs/dummy_pipeline.yaml`
- `configs/dummy_pipeline_banana.yaml`

这个脚本适合后续扩展为批量实验入口。

### `scripts/summarize_dummy_results.py`

实验结果汇总脚本。

它会扫描：

```text
outputs/dummy_pipeline/*/result.json
```

然后提取关键字段，生成：

```text
outputs/dummy_pipeline_summary.csv
```

这个 CSV 可以用于后续实验分析、表格展示和结果对比。

### `src/world_vla_graspqa/perception/dummy_perception.py`

Dummy 感知模块。

当前不读取真实图片，而是直接从配置文件中读取物体列表。

后续可以替换为：

- VLM 场景理解模块。
- 目标检测模块。
- RGB-D 感知模块。
- 场景图构建模块。

### `src/world_vla_graspqa/graspqa/dummy_graspqa.py`

Dummy 抓取问答模块。

当前根据任务指令和物体列表，选择应该抓取的目标物体。

例如：

```text
Pick up the red cube and place it into the blue bowl.
```

会选择：

```text
red cube
```

后续可以替换为基于 VLM 的抓取问答模块。

### `src/world_vla_graspqa/action/dummy_action_generator.py`

候选动作生成模块。

当前为目标物体生成三个固定候选抓取姿态：

- `top_down`
- `left_side`
- `right_side`

后续可以替换为真实抓取候选生成器，例如基于深度图、点云或 GraspNet 风格的方法。

### `src/world_vla_graspqa/world_model/dummy_world_model.py`

Dummy 世界模型模块。

当前根据抓取姿态给出固定成功率：

```text
top_down   → 0.85
left_side  → 0.65
right_side → 0.60
```

后续可以替换为可学习的世界模型，用于预测动作后果、成功概率、失败原因或下一状态。

### `src/world_vla_graspqa/planner/dummy_planner.py`

规划器模块。

当前策略很简单：选择 `predicted_success` 最高的动作。

后续可以扩展为闭环规划器：

```text
观察 → 规划 → 执行动作 → 观察结果 → 判断失败 → 重新规划
```

### `src/world_vla_graspqa/utils/config.py`

配置读取工具。

当前用于读取 YAML 配置文件。

### `src/world_vla_graspqa/utils/io.py`

输入输出工具。

当前包含：

- 创建目录。
- 创建时间戳 run 目录。
- 保存 JSON。
- 保存 YAML。

### `src/world_vla_graspqa/utils/summary.py`

结果汇总工具。

当前包含：

- 读取 result.json。
- 搜索所有实验结果。
- 将单个实验结果转成 CSV 行。
- 写入 summary CSV。

### `src/world_vla_graspqa/utils/logger.py`

日志打印工具。

当前提供统一格式的日志输出，例如：

```text
[Perception] Detected objects: red cube, blue bowl, yellow banana
```

## 开发路线

### Stage 1：工程基础

- [x] 创建 GitHub 仓库和项目结构
- [x] 配置 Conda 环境
- [x] 搭建 dummy 端到端 pipeline
- [x] 添加 YAML 配置系统
- [x] 添加时间戳实验输出目录
- [x] 添加批量运行脚本
- [x] 添加结果汇总脚本
- [x] 添加单元测试
- [x] 添加 `ruff` 和 `black` 代码检查

### Stage 2：数据与场景接口

- [ ] 添加样例桌面场景图片
- [ ] 定义场景数据格式
- [ ] 添加图像加载工具
- [ ] 添加物体标注格式
- [ ] 将 perception 模块连接到真实图像

### Stage 3：VLM 感知与 GraspQA

- [ ] 添加 VLM 场景解析模块
- [ ] 设计 prompt 模板
- [ ] 生成物体级场景描述
- [ ] 生成抓取相关问答结果

### Stage 4：Mini World Model

- [ ] 定义 action-outcome 数据格式
- [ ] 训练轻量级结果预测模型
- [ ] 对比世界模型排序、随机策略和规则策略

### Stage 5：闭环规划

- [ ] 添加执行反馈
- [ ] 添加失败检测
- [ ] 添加重新规划循环
- [ ] 添加 memory 和 critic 模块

## License

MIT License.