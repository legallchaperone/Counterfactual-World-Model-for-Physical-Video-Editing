# E2W STATUS

当前项目状态真源是根目录 `STATUS.md`。

本文件只保留摘要，避免 `docs/STATUS.md` 和根 `STATUS.md` 分叉。

最后同步：2026-06-16 UTC。

## 摘要

- 当前接口已统一为 `tools/e2w_remove.py` / `tools/e2w_add.py` / `tools/e2w_add_then_remove.py`，共享 `tools/e2w_pipeline_core.py`。
- 固定跑法见 `docs/UNIFIED_PIPELINE_RUNBOOK.md`。
- 旧入口 `run_counterfactual_planner_pipeline.py` / `run_add_pipeline_interface.py` / `run_add_then_remove_pipeline.py` 只保留为 shim，不再作为新文档或新自动化入口。
- Counterfactual Planner 是当前 remove 设计；兼容 schema id 为 `e2w.planner_output.v8_tool_augmented_grounding.v1`。
- Add 是一等公民：add planner -> planner-region masked inpaint -> SAM2-on-edited-first-frame -> VACE(add)。
- 当前统一接口完成的是 current-spec INTERFACE/STRUCTURAL 对齐；CONTROL/VISUAL/RESEARCH 证据仍未建立。

详细状态、证据路径和 next actions 见 `../STATUS.md`。
