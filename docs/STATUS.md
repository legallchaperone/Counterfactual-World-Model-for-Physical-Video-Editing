# E2W STATUS

当前项目状态真源是根目录 `STATUS.md`。

本文件只保留摘要，避免 `docs/STATUS.md` 和根 `STATUS.md` 分叉。

最后同步：2026-06-08 UTC。

## 摘要

- v0.2/v7 executable planner: 最新 remove8 planner gate 是 1/8 ok，仍不能跑 full forward pass。
- v8 planner-text experiment: `vlm_planner_lora_v8_20260604_v3` 在 seed_v3 eval 34 条上 34/34 JSON parse、33/34 schema/`if_removed` pass；valid fill types 为 `background_continuation` 27、`occlusion_reveal` 6，但不输出 executable quadmask grounding。
- VACE Phase 1A control-branch 数据路线与 planner 训练路线独立，训练阶段不互相依赖。

详细状态、证据路径和 next actions 见 `../STATUS.md`。
