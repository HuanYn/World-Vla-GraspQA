# World-Vla-GraspQA
基于世界模型的机器人抓取问答与闭环操作系统

## Quick Start

Create and activate the Conda environment:

```bash
conda create -n wvg python=3.10 -y
conda activate wvg
pip install -r requirements.txt
pip install -e .
```

Run the dummy end-to-end pipeline:

```bash
python scripts/run_dummy_pipeline.py
```

Expected output:

```text
[Observation] Loaded dummy scene.
[Instruction] Pick up the red cube and place it into the blue bowl.
[Perception] Detected objects: red cube, blue bowl, yellow banana
[GraspQA] Question: Which object should the robot grasp?
[GraspQA] Answer: red cube
[Action] Generated 3 candidate actions.
[WorldModel] Predicted outcomes for candidate actions.
[Planner] Selected best action: grasp(red cube)
[Result] Dummy pipeline finished successfully. Best action=grasp(red cube), pose=top_down, predicted_success=0.85
```

Run tests and code checks:

```bash
pytest
ruff check .
black --check .
```

Current local milestone:

- Built a minimal runnable pipeline.
- Added dummy perception, GraspQA, action generation, world model, and planner modules.
- Added unit tests for config loading, GraspQA, world model, and planner.
- Added editable package setup with `pip install -e .`.
