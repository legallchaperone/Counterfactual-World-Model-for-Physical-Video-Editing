# E2W 当前状态

最后更新：2026-06-04 UTC。

## 当前目标

使用 SFT VLM planner，对 8 个 remove smoke samples 跑一次 fresh artifact-first forward pass。在 planner stage 通过 strict gates 之前，不要继续到 mask、Qwen first-frame edit、VACE、package/report 或 freshness audit。

## 当前 Planner 状态

当前 train/eval SFT JSONL 已同步为 final-rule-only prompts：

- train: `/data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_train_v6_teacher_grounded.jsonl`
- eval: `/data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl`

数据计数：

- train rows: 262
- eval rows: 30
- executable assistant validation: train 262/262, eval 30/30

当前有效的数据 caveat：

- executable quadmask labels 能 validate，但不是所有 assistant labels 都满足 target-free `counterfactual_expectation.if_removed` contract。
- 这意味着当前 training targets 和 VACE-facing text 的 prompt rule 不一致。
- 完成 label cleanup 后，把这一段替换成：`Labels have passed the serialize_vace_prompt() gate; train/eval are fully compliant.`

## 当前 Checkpoints

最新 final-rule LoRA：

- `/data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v6_executable_final_rules_20260603`

训练摘要：

- steps: 68/68
- train loss: 0.72217
- eval loss: 0.70221
- epoch: 1.03

base model comparison 使用 `tools/eval_vlm_planner.py --no-adapter` 跑过。

## 最新 Planner Runs

Final-rule LoRA 30 eval：

- run: `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_eval30_20260603T1029Z`
- parse/schema/operation: 30/30
- affected grid valid: 30/30
- quadmask executable: 19/30
- final statuses: ok 3/30, VACE prompt contract failed 16/30, planner quadmask failed 11/30

Final-rule LoRA 8 remove smoke：

- run: `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_remove8_20260603T1038Z`
- parse/schema/operation: 8/8
- affected grid valid: 8/8
- quadmask executable: 5/8
- final ok: 0/8

Base model 8 remove smoke：

- run: `/data/cwx/E2W/runs/e2w_v6_base_planner_remove8_20260603TbaseZ`
- parse/schema/operation: 8/8
- affected grid valid: 8/8
- quadmask executable: 2/8
- final ok: 1/8

咨询用 handoff summary：

- `docs/vlm_planner_handoff_20260603.md`

## 当前阻塞点

当前 blocker 是数据和 contract 不匹配，不是旧 schema support 问题。

planner 能产出 top-level v6 JSON，也经常能产出 executable grounding。它仍然失败，是因为 SFT labels 中包含 target-contaminated remove `if_removed` text，而 prompt 和 runtime contract 要求 `if_removed` target-free。

典型 problematic label/output patterns：

- `<target> is no longer present`
- `no <target>`
- `<target> does not exist`
- `where the mug was`
- target subparts/materials in VACE-facing text

不要从当前 planner checkpoint 运行 full forward pass。

## Next Actions

- [ ] 1. 给 train/eval assistant labels 加 label-level `serialize_vace_prompt()` gate。
- [ ] 2. 重写或重新生成所有不满足 VACE prompt contract 的 remove `counterfactual_expectation.if_removed`。
- [x] 3. 保持 final-rule-only planner prompt；除非 prompt contract 变化，否则无需操作。
- [ ] 4. label cleanup 后重新训练。
- [ ] 5. 重新评估 30 eval 和 8 smoke。
- [ ] 6. planner 通过后，才运行 mask、Qwen first-frame edit、VACE、package/report、freshness audit。

## Reference Only

已知 downstream interface reference run：

- `/data/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z`

只把它当作 downstream interface reference。它不能证明当前 strict SFT VLM planner 通过。
