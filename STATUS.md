# E2W 当前状态

最后更新：2026-06-08 UTC。

本文件是当前项目状态真源。`docs/STATUS.md` 只保留指向本文件的摘要；不要把历史 handoff、README 示例或旧 run 当作当前状态。

## 总结

当前不是可以直接 full forward pass 的状态。

- v0.2/v7 executable planner 路线：remove8 gate 仍未通过，最新 fresh planner run 是 1/8 ok。
- v8 planner-text 路线：已完成 image-only/tool-augmented schema 实验和 seed_v3 LoRA v3 eval，34 eval 中 33/34 schema/`if_removed` pass；它不输出 quadmask grounding，不能替代 v0.2 full pipeline。
- VACE Phase 1A control-branch 数据路线与 planner 训练路线独立，训练阶段不互相依赖。

planner strict gates 通过前，不要继续 mask、Qwen first-frame edit、VACE、package/report 或 freshness audit。

## 线 A - Planner 训练和推理状态

### v0.2/v7 executable planner

目标仍是让 SFT VLM planner 对 8 个 remove smoke samples 产出可执行 quadmask grounding 和 target-free VACE-facing text。

最新 v7 数据/checkpoint：

- data: `/data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v7_targetfree_final.jsonl`
- checkpoint: `/data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v7_targetfree_final_20260604`
- run: `/data/cwx/E2W/runs/e2w_v7_targetfree_final_planner_remove8_20260604T0904Z`

最新 v7 remove8 结果：

- parse/schema/operation: 8/8
- affected grid valid: 8/8
- primary bbox valid: 7/8
- quadmask executable: 7/8
- final ok: 1/8
- failure shape:
  - 6/8 `vace_prompt_contract_failed`
  - 1/8 `planner_quadmask_failed` (`0056`, `missing_primary_bbox`)
  - 1/8 `ok` (`0077`)

结论：v7 已经明显改善旧 v6 的 schema/grounding，但 remove VACE-facing text 仍常生成 target aliases、target subparts/materials 或 negative wording，例如 `without ...`、`no longer present`。这仍必须 hard fail，不能静默删除句子或用 neutral fallback。

### v8 planner-text experiment

v8 是 image-only/tool-augmented planner text 实验，schema 是 `e2w.planner_output.v8_tool_augmented_grounding.v1`。它拆分 `counterfactual_state` 为 surface/lighting/shadow/temporal/interaction/geometry，并生成 target-free `if_removed`。

v8 不输出 executable quadmask spec，不是当前 v0.2 smoke chain 的 drop-in replacement。

最新 v8 checkpoints/evals：

- failed first run: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604`
  - eval: 0/30 JSON parse, 0/30 schema valid
  - failure: model answered with prose/instructions instead of JSON
- current v2 run: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v2`
  - train: 300 steps, epoch 10.0
  - train loss: 1.32466
  - eval loss: 1.24883
  - eval output: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v2/eval_v8_outputs.jsonl`
  - eval summary: 30/30 JSON parse, 24/30 schema valid, 24/30 `if_removed` pass
- current v3 seed_v3 run: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3`
  - data: `/data/cwx/E2W/data/line_c_annotations/seed_v3.jsonl`
  - split: train 136, eval 34
  - train: 300 steps, epoch 8.8235
  - train loss: 1.33311
  - eval loss: 1.23665
  - eval output: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3/eval_v8_outputs.jsonl`
  - eval summary: 34/34 JSON parse, 33/34 schema valid, 33/34 `if_removed` pass
  - valid fill_type distribution: `background_continuation` 27, `occlusion_reveal` 6
  - details: `docs/experiments/A12_seed_v3_v8_v3_20260608.md`

结论：v8 证明 prompt/schema redesign 可以显著提升 target-free text compliance。seed_v3/v3 也恢复了一部分 `background_continuation` 以外的 fill_type 多样性，但 v8 仍未接回 executable grounding、quadmask、Qwen first-frame edit 或 VACE full forward path。

## 线 B - VACE Phase 1A Control Branch

该路线独立于 planner SFT 训练，目标是训练 VACE 侧的 gated causal control branch。

截至 2026-06-04 的记录：

- B1 下载 80 条 Pexels raw videos: `8de58bb`
- B2 标准化 80 条候选背景 `[81,480,832,3]`: `4b234e7`
- B3 筛选 16 条 clean backgrounds，58/80 自动通过: `fc39ad0`
- B4 生成 SI-A/B/C composites，12 个，0 audit failure: `b121c51`
  - train manifest 16 行，8 add + 8 remove
  - eval manifest 8 行，4 add + 4 remove
- B5 后续状态需要从 control-branch 工作树或对应 artifacts 重新确认。

线 A 和线 B 只在推理时汇合：

```text
用户指令 -> planner -> quadmask + VACE prompt -> VACE -> edited video
```

## 当前阻塞点

主要 blocker 不是旧 schema support 问题，而是 planner route 尚未同时满足两个下游 gate：

1. v0.2/v7 route 仍会生成 target-contaminated or removal-residue VACE-facing text。
2. v8 route 改善了 target-free text，但暂不产出 executable quadmask grounding。

因此当前不要从任何 planner checkpoint 直接运行 full forward pass。

## Next Actions

- [ ] 1. 选择主线：继续修 v0.2/v7 executable planner，或把 v8 text schema 接回 executable grounding。
- [ ] 2. 如果继续 v0.2/v7：补充/重训正向描述反事实场景的 labels，避免 `without/no longer present/where X was` 类表达。
- [ ] 3. 如果推进 v8：设计 v8 -> executable quadmask grounding 的接口，或拆成 text planner + grounding planner 两阶段。
- [ ] 4. 重新评估 30 eval 和 8 remove smoke。
- [ ] 5. 只有 planner gate 全部通过后，才运行 mask、Qwen first-frame edit、VACE、package/report、freshness audit。

## Historical References

这些 artifacts 只能用于历史对比或 downstream interface reference，不能证明当前 strict planner 通过：

- `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_eval30_20260603T1029Z`
- `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_remove8_20260603T1038Z`
- `/data/cwx/E2W/runs/e2w_v6_base_planner_remove8_20260603TbaseZ`
- `/data/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z`
- `docs/vlm_planner_handoff_20260603.md`
