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
