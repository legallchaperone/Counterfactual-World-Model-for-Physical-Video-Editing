# Agent Instructions

本仓库是 artifact-first E2W smoke pipeline 和 v0.3 quadmask VACE contract 的 canonical code/docs/tests root。

当前项目状态请读取 `STATUS.md`。不要从本文件推断当前状态。

## 范围和根目录

- 代码、测试、文档、Git 都从 `/home/cwx/E2W` 工作。
- 持久数据、checkpoint、run、output、external assets、HF cache links、tmp roots 使用 `/data/cwx/E2W`。
- full-run artifacts 写到 `/data/cwx/E2W/runs`。
- 使用 `/data/cwx/conda/envs/edit2world-phase1-real/bin/python`；不要假设 bare `python` 有正确依赖。
- 不要 kill 其他用户的 GPU jobs。真实 CUDA run 前先检查 GPU contention：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

## Agent 行为规则

- 优先使用本地真实命令、日志、manifest、report 和代码证据，不要泛泛解释。
- 不要为了让 run 通过而削弱、删除或绕过 schema、prompt、parser、reporting tests。
- 如果 `tests/test_v02_contracts.py` 失败，修 data、prompt、parser、planner 或 runner contract，不要改弱测试。
- planner failure 必须从 artifact 归因，例如 `manifest.jsonl`、`planner_pred/*/planner_eval.json`、`raw_output.txt`、`raw.pred.json`、`report.md`、`artifact_freshness.json`。
- interface success 不是 visual success。不要用文件大小、black-frame check 或 interface check 推断人工视觉质量。
- 除非人工已经检查 artifact，否则保持 `qwen_visual_review_status` 为 `unreviewed`。

## 稳定合约

- canonical planner I/O schema 是 `e2w.planner_io.v6_executable.v1`。
- planner train/eval/smoke prompts 必须使用 `tools/e2w_v0_common.py::build_planner_user_prompt`。
- canonical planner prompt 是 final-rule-only：role/schema、task operation、full schema JSON、当前 `video_id`/request、最后是 final rules。
- 不要在 planner prompt 里嵌入 one-shot wrapper examples，除非 output format 被重新设计并验证过。
- 旧的空 `{"quadmask_spec": {"primary": {}, "affected": {}, "keep": {}}}` schema 只允许 archive 使用。
- planner output 必须是一个完整 top-level JSON object，并且每个 selected sample 必须写 `raw_output.txt` 和 `raw.pred.json`。
- executable quadmask grounding 需要 `primary.keyframes[].bbox_xyxy_norm1000`、`primary.keyframes[].positive_points_norm1000`、`affected.grid_shape`、`affected.frame_ranges[].cells`。
- remove operation 中，`counterfactual_expectation.if_removed` 是 VACE-facing，必须 target-free。
- VACE prompts 不能包含 target aliases、visible target subparts/materials，也不能包含 `remove`、`delete`、`erase`、`removed`、`no <target>`、`without <target>` 等 removal wording。
- 如果 planner text 违反 VACE prompt contract，必须 hard fail。不要静默删除句子，也不要用 neutral fallback。
- Qwen-Image-Edit 只消费 image + text prompt；target masks 是 QC/debug artifacts，metadata 记录 `target_mask_consumed_by_backend: false`。
- v0.3 VACE 消费 `quadmask.npy`，取值 `0/63/127/255` 分别表示 primary、primary+affected overlap、affected、keep。
- v0.3 不代表已经学到 quadmask semantics，除非明确记录了 trained adapter 或 checkpoint。

## Validation

需要验证 pipeline changes 时运行 static checks：

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py
```
