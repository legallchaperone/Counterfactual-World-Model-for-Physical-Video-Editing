# E2W STATUS

当前项目状态真源是根目录 `STATUS.md`。

本文件只保留摘要，避免 `docs/STATUS.md` 和根 `STATUS.md` 分叉。

最后同步：2026-06-09 UTC。

## 摘要

- v8 planner 是当前正确设计：`e2w.planner_output.v8_tool_augmented_grounding.v1`。
- v8 seed_v3 eval：34/34 JSON parse、33/34 schema/`if_removed` pass；valid fill types 为 `background_continuation` 27、`occlusion_reveal` 6。
- 当前 blocker 是 v8 grounding bridge / runtime adapter 尚未按 current spec 完成结构性对齐。
- v6/v7/v0.2 executable-planner artifacts 已归档为历史参考，不再作为 current planner baseline。
- VACE Phase 1A control-branch 数据路线与 planner 训练路线独立，训练阶段不互相依赖。

详细状态、证据路径和 next actions 见 `../STATUS.md`。
