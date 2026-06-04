# E2W STATUS

## 当前状态（2026-06-04）

### 线 A — Planner 训练（main branch）

- 数据：`v7_targetfree_final`，train 262 行 / eval 30 行，全部通过 validator。
- 最新 checkpoint：`vlm_planner_lora_physics_iq_v7_targetfree_final_20260604`。
- Gate 结果：remove8 1/8 pass。
- 实验 A6/A7：同模型自改写实验，3/7 pass；但 A7 显示模型无法遵从“禁止负向陈述”指令，始终输出 `with no X` 句式。
- 结论：prompt 工程到头，根本原因是训练数据未覆盖正向描述反事实场景的写法。
- 待做 A8：补充人工写的正向 gold label 训练数据。
- 详细实验记录：`docs/experiments/A6_self_correction_20260604.md`。

### 线 B — VACE Phase 1A 数据（feat/phase1-v04-control-branch）

- B1 下载 80 条 Pexels raw videos — `8de58bb`。
- B2 标准化 80 条候选背景 `[81,480,832,3]` — `4b234e7`。
- B3 筛选 16 条 clean backgrounds（58/80 自动通过）— `fc39ad0`。
- B4 生成 SI-A/B/C composites，12 个，0 audit failure — `b121c51`。
  - train manifest 16 行（8 add + 8 remove）。
  - eval manifest 8 行（4 add + 4 remove）。
- B5 进行中：audit + `overfit_16.jsonl` + contact sheet。

### 架构说明

线 A 和线 B 完全独立，只在推理时汇合：

```text
用户指令 → planner → quadmask+VACE prompt → VACE → 编辑视频
```

训练阶段互不依赖。
