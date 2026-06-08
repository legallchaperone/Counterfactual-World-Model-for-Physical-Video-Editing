# E2W VLM Planner Handoff - 2026-06-03

## Context

Repo:

- code/docs/tests: `/home/cwx/E2W`
- durable data/checkpoints/runs: `/data/cwx/E2W`
- Python env: `/data/cwx/conda/envs/edit2world-phase1-real/bin/python`

Goal:

- Get the SFT VLM planner to pass the strict planner gate for the 8 remove smoke samples.
- Only after planner gate passes should the pipeline continue to quadmask build, Qwen first-frame edit, VACE, package/report, and freshness audit.

Current strict requirements:

- Planner must write `raw_output.txt` and `raw.pred.json` for every sample.
- Planner output must be complete top-level JSON in schema `e2w.planner_io.v6_executable.v1`.
- Planner must produce executable quadmask grounding:
  - `primary.keyframes[].bbox_xyxy_norm1000`
  - `primary.keyframes[].positive_points_norm1000`
  - `affected.grid_shape`
  - `affected.frame_ranges[].cells`
- For remove operations, `counterfactual_expectation.if_removed` is copied into the VACE prompt, so it must be target-free.
- Remove `if_removed` must not name the removed target, aliases, visible target parts/materials, or negative forms such as `no <target>`, `without <target>`, `<target> is removed`, or `<target> is no longer present`.

## Current Canonical Prompt

Canonical prompt generator:

- `tools/e2w_v0_common.py::build_planner_user_prompt`

Prompt style now kept:

- final-rule-only prompt
- no one-shot examples
- no wrapper examples
- full schema JSON first
- current `video_id` and user request
- final rules at the end

Reason:

- One-shot prompt experiments caused the current LoRA to copy wrapper keys such as `input_video_id`, `input_user_request`, `good_complete_output`, and `output_excerpt`, breaking strict top-level JSON parsing.
- Final-rule-only preserves parse/schema stability.

Docs updated:

- `docs/CONTRACT.md`
- `README.md`
- `docs/v0_3_vace_quadmask_training_status.md`

Code changed:

- `tools/e2w_v0_common.py`: final-rule-only canonical prompt.
- `tools/eval_vlm_planner.py`: added `--no-adapter` to run base model without LoRA for comparison.

## Data Status

Train/eval prompts were synchronized to the final-rule-only prompt.

Train JSONL:

- `/data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_train_v6_teacher_grounded.jsonl`
- rows: 262
- executable assistant validation: 262/262
- archive:
  `/data/cwx/E2W/data/physics_iq_vlm_sft/archive/v6_prompt_final_rules_only_20260603_train_eval_sync/vlm_planner_sft_train_v6_teacher_grounded.jsonl.20260603T102431Z.old_schema`

Eval JSONL:

- `/data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl`
- rows: 30
- executable assistant validation: 30/30
- archive:
  `/data/cwx/E2W/data/physics_iq_vlm_sft/archive/v6_prompt_final_rules_only_20260603_train_eval_sync/vlm_planner_sft_eval_v6_teacher_grounded.jsonl.20260603T102432Z.old_schema`

Earlier train grounding generation:

- original train rows: 270
- grounded accepted rows: 262
- rejected/review rows: 8
- API retry rows: 3/3 succeeded and were included

## Retrained Final-Rule LoRA

Checkpoint:

- `/data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v6_executable_final_rules_20260603`

Training command:

```bash
cd /home/cwx/E2W
CUDA_VISIBLE_DEVICES=4 /data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/train_qwen25vl_lora_sft.py \
  --train-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_train_v6_teacher_grounded.jsonl \
  --eval-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl \
  --base-model /data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct \
  --output-dir /data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v6_executable_final_rules_20260603 \
  --max-steps 68 \
  --save-steps 68 \
  --eval-steps 68
```

Training result:

- completed: 68/68 steps
- train rows: 262
- eval rows: 30
- train runtime: 157.3 s
- train loss: `0.72217`
- eval loss: `0.70221`
- epoch: `1.03`

## Planner Evaluation Runs

Final-rule LoRA 30 eval:

- `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_eval30_20260603T1029Z`

Summary:

- `json_parse_ok`: 30/30
- `schema_valid`: 30/30
- `operation_accuracy`: 30/30
- `affected_grid_valid`: 30/30
- `primary_bbox_valid`: 19/30
- `quadmask_spec_executable`: 19/30
- final status:
  - `ok`: 3/30
  - `vace_prompt_contract_failed`: 16/30
  - `planner_quadmask_failed`: 11/30

Final-rule LoRA 8 remove smoke:

- `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_remove8_20260603T1038Z`

Summary:

- `json_parse_ok`: 8/8
- `schema_valid`: 8/8
- `operation_accuracy`: 8/8
- `affected_grid_valid`: 8/8
- `primary_bbox_valid`: 5/8
- `quadmask_spec_executable`: 5/8
- final `ok`: 0/8

Base model 8 remove smoke:

- `/data/cwx/E2W/runs/e2w_v6_base_planner_remove8_20260603TbaseZ`
- command used `tools/eval_vlm_planner.py --no-adapter`

Summary:

- `json_parse_ok`: 8/8
- `schema_valid`: 8/8
- `operation_accuracy`: 8/8
- `affected_grid_valid`: 8/8
- `primary_bbox_valid`: 2/8
- `quadmask_spec_executable`: 2/8
- final `ok`: 1/8

## Base vs LoRA Comparison on 8 Smoke

| metric | base-only | final-rule LoRA |
|---|---:|---:|
| `json_parse_ok` | 8/8 | 8/8 |
| `schema_valid` | 8/8 | 8/8 |
| `operation_accuracy` | 8/8 | 8/8 |
| `affected_grid_valid` | 8/8 | 8/8 |
| `primary_bbox_valid` | 2/8 | 5/8 |
| `quadmask_spec_executable` | 2/8 | 5/8 |
| final `ok` | 1/8 | 0/8 |

Interpretation:

- LoRA improves executable grounding versus base model.
- Base model has many bbox outputs as normalized floats like `[0.5, 0.5, 0.7, 0.7]` instead of norm1000.
- LoRA still fails target-free `if_removed` badly.
- Neither base nor LoRA is ready for full forward pass.

## 8-Smoke Failure Details

Final-rule LoRA run:

- `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_remove8_20260603T1038Z`

| sample | final-rule LoRA status | cause |
|---|---|---|
| `0052` | `vace_prompt_contract_failed` | `if_removed`: `The Newton's cradle is now empty, with no metal balls hanging from the frame.` |
| `0056` | `planner_quadmask_failed` | zero-area bbox `[400, 300, 400, 300]`; text also target-contaminated |
| `0070` | `vace_prompt_contract_failed` | names `clear glass with water inside`; also says where the glass was located |
| `0076` | `vace_prompt_contract_failed` | names `yellow mug`; also says `shadow of the mug` |
| `0077` | `planner_quadmask_failed` | zero-area bbox `[500, 300, 500, 300]`; target descriptor contamination |
| `0112` | `vace_prompt_contract_failed` | names `black balloon` |
| `0128` | `vace_prompt_contract_failed` | says `free of the shallow dish of light blue liquid` |
| `0341` | `planner_quadmask_failed` | zero-area bbox `[500, 500, 500, 500]`; text also target-contaminated |

Base model run:

- `/data/cwx/E2W/runs/e2w_v6_base_planner_remove8_20260603TbaseZ`

| sample | base status | note |
|---|---|---|
| `0052` | `ok` | target-free enough and executable bbox |
| `0056` | `planner_quadmask_failed` | bbox as floats `[0.4, 0.3, 0.6, 0.5]` |
| `0070` | `vace_prompt_contract_failed` | target-contaminated |
| `0076` | `planner_quadmask_failed` | zero/floating bbox `[0.5, 0.5, 0.5, 0.5]`; target-contaminated |
| `0077` | `planner_quadmask_failed` | zero/floating bbox `[0.5, 0.5, 0.5, 0.5]`; target-contaminated |
| `0112` | `planner_quadmask_failed` | bbox floats `[0.5, 0.5, 0.7, 0.7]`; VACE text clean |
| `0128` | `planner_quadmask_failed` | bbox floats `[0.5, 0.5, 0.7, 0.7]`; target-contaminated |
| `0341` | `planner_quadmask_failed` | bbox floats `[0.5, 0.5, 0.9, 0.9]`; target-contaminated |

## Important Observation

The final-rule LoRA is not failing because it cannot parse the prompt or output schema.

It is good at:

- top-level JSON parsing
- v6 schema shape
- operation selection
- affected grid fields
- positive points

It is still bad at:

- target-free `counterfactual_expectation.if_removed`
- valid bbox geometry for some samples

The target-free problem likely requires data/label supervision, not more inference prompt tweaks.

## Prompt Experiments Already Tried

Old v6 prompt:

- parse/schema stable
- executable improved over old v5
- target contamination remained

One-shot excerpt prompt:

- run: `/data/cwx/E2W/runs/e2w_v6_planner_remove8_prompt_oneshot_20260603T0846Z`
- outcome: 5/8 parse/schema, 1/8 ok
- problem: model copied example/excerpt style and omitted required fields

Complete one-shot prompt:

- run: `/data/cwx/E2W/runs/e2w_v6_planner_remove8_prompt_complete_oneshot_20260603T0923Z`
- outcome: 0/8 parse/schema
- problem: model copied wrapper keys `input_video_id`, `input_user_request`, `good_complete_output`

Final-rule-only prompt without retrain:

- run: `/data/cwx/E2W/runs/e2w_v6_planner_remove8_prompt_final_rules_only_20260603T1016Z`
- outcome: parse/schema 8/8, executable 5/8, ok 1/8
- problem: target contamination remained

Final-rule-only prompt with retrain:

- checkpoint: `/data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v6_executable_final_rules_20260603`
- 8-smoke outcome: parse/schema 8/8, executable 5/8, ok 0/8
- problem: target contamination remained and no full smoke pass

## Questions for External Consultation

1. Should `counterfactual_expectation.if_removed` be removed from the VLM planner output entirely and generated by a deterministic postprocessor from non-target effects?
2. Should target-free VACE text be trained as a separate field with many positive/negative contrastive examples instead of relying on one `if_removed` instruction?
3. Should we add a validator-in-the-loop relabel step that rejects any assistant label whose `if_removed` fails `serialize_vace_prompt()` before SFT?
4. Should bbox be supervised with a separate spatial-only head/data path, since current LoRA still outputs zero-area boxes or normalized floats?
5. Is it better to split planner into two models/stages: semantic plan then spatial grounding?
6. Should `if_removed` be allowed to mention target-adjacent effects such as `mug shadow`, or should target parts/materials/descriptors remain strictly banned as currently enforced?
7. Is the current eval too strict for VACE prompt generation, or is hard failure correct because target terms directly leak into the video generation prompt?

## Recommended Next Step

Do not run mask/Qwen/VACE from the latest planner checkpoint.

Most useful next intervention:

1. Rewrite or regenerate train labels so every remove `if_removed` passes `serialize_vace_prompt()`.
2. Add a label-level gate before training: assistant labels with target-contaminated `if_removed` cannot enter train/eval.
3. Keep final-rule-only prompt.
4. Retrain.
5. Re-evaluate 30 eval and 8 smoke.

