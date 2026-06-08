# World-Vla-GraspQA Architecture

本文档记录 World-Vla-GraspQA 当前阶段的系统架构、模块职责、数据流和后续可替换接口。

## 当前最小闭环

```text
image + annotation
  ↓
scene description
  ↓
GraspQA prompt
  ↓
Dummy VLM / GraspQA
  ↓
target_object
  ↓
candidate action generation
  ↓
world model prediction
  ↓
planner
  ↓
executor
  ↓
critic
  ↓
closed-loop trace
  ↓
feedback action-outcome records
]633;E;echo "";a48919bb-049a-4f23-8013-bb532e44e1ca]633;C
## Learned Mini World Model

当前项目已支持学习型 Mini World Model。

数据来源：

```text
data/action_outcomes/dummy_action_outcomes.json
outputs/action_outcomes/feedback_records.jsonl
```

训练数据输出：

```text
outputs/world_model_training/training_samples.jsonl
```

模型 checkpoint：

```text
outputs/world_model_training/logistic_world_model.pkl
```

当前 LearnedWorldModel 使用 Logistic Regression，根据 symbolic features 预测动作成功率。

当前 world model 支持三种模式：

```text
dummy
empirical
learned
```

配置示例：

```yaml
world_model:
  mode: learned
  model_path: outputs/world_model_training/logistic_world_model.pkl
  success_threshold: 0.5
```

三方对比脚本：

```bash
python scripts/run_world_model_three_way_comparison.py
```

