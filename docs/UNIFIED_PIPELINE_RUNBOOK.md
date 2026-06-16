# Unified E2W Pipeline Runbook

Last updated: 2026-06-16

This is the fixed runbook for the current unified E2W interfaces. The current
runtime/spec source is still `docs/E2W_SPEC.md`; this file only records how to run
that contract.

## Canonical entry points

Use only these three current entry points for new runs:

| Operation | Current entry point | Notes |
|---|---|---|
| remove | `tools/e2w_remove.py` | Counterfactual Planner -> GroundingDINO/SAM2 -> first-frame edit -> VACE remove |
| add | `tools/e2w_add.py` | Add planner -> masked inpaint first-frame edit -> SAM2 on edited first frame -> VACE add |
| add then remove | `tools/e2w_add_then_remove.py` | Orchestrates `e2w_add.py` then `e2w_remove.py` |

All three share `tools/e2w_pipeline_core.py` for planner inference, first-frame
conditioning-video construction, generation masks, VACE invocation, and runtime
metadata.

Deprecated compatibility shims only:

- `tools/run_counterfactual_planner_pipeline.py` -> `tools/e2w_remove.py`
- `tools/run_add_pipeline_interface.py` -> `tools/e2w_add.py`
- `tools/run_add_then_remove_pipeline.py` -> `tools/e2w_add_then_remove.py`

Do not use the shim names in new docs, reports, or automation.

## Common environment

```bash
cd /home/cwx/E2W
export PY=/data/cwx/conda/envs/edit2world-phase1-real/bin/python
export ADD_PLANNER=/data/cwx/E2W/checkpoints/vlm_planner_lora_add_v1_20260615
export CONTROL_BRANCH=/data/cwx/E2W/checkpoints/v04_real_overfit_14b_specfix_selfinsert_20260612
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Before real CUDA runs, check GPU contention and do not kill other users' jobs:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

## Remove run

Input is an eval JSONL row containing the original frame/image and a remove request.
`e2w_remove.py` uses the current Counterfactual Planner adapter by default.

```bash
CUDA_VISIBLE_DEVICES=<gpu> $PY tools/e2w_remove.py \
  --eval-jsonl <remove_eval.jsonl> \
  --output-dir /data/cwx/E2W/runs/e2w_remove_<sample_or_batch_id> \
  --sample-count 1 \
  --seed 0 \
  --control-branch-checkpoint "$CONTROL_BRANCH" \
  --vace-sample-steps 8
```

Structural debug only, not acceptance:

```bash
CUDA_VISIBLE_DEVICES=<gpu> $PY tools/e2w_remove.py \
  --eval-jsonl <remove_eval.jsonl> \
  --output-dir /data/cwx/E2W/runs/e2w_remove_skipvace_<id> \
  --sample-count 1 \
  --seed 0 \
  --skip-vace
```

## Add run

The add planner adapter is required explicitly; there is no archived add default.
`--frame-num` is optional. If omitted, the runner uses the source video length rounded
to a Wan-compatible `4n+1` value and records the alignment metadata.

```bash
CUDA_VISIBLE_DEVICES=<gpu> $PY tools/e2w_add.py \
  --source-video <source_video.mp4> \
  --user-prompt "Add a red mug on the table near the center of the image." \
  --sample-id <sample_id> \
  --run-dir /data/cwx/E2W/runs/e2w_add_<sample_id> \
  --planner-adapter "$ADD_PLANNER" \
  --control-branch-checkpoint "$CONTROL_BRANCH" \
  --planner-attempts 3 \
  --qwen-steps 20 \
  --vace-sample-steps 8 \
  --cuda-visible-devices <gpu>
```

Fast interface smoke settings may lower `--qwen-steps` and `--vace-sample-steps`,
but the result remains INTERFACE evidence only unless separately reviewed/tested.

## Add-then-remove run

This is the fixed chained run for visual-candidate debugging. It first runs the add
interface, extracts frames from the add output, then runs the remove interface on the
newly added target from add-stage metadata.

```bash
CUDA_VISIBLE_DEVICES=<gpu> $PY tools/e2w_add_then_remove.py \
  --source-video <source_video.mp4> \
  --add-prompt "Add a red mug on the table near the center of the image." \
  --sample-id <sample_id> \
  --run-dir /data/cwx/E2W/runs/add_then_remove/<sample_id> \
  --add-planner-adapter "$ADD_PLANNER" \
  --control-branch-checkpoint "$CONTROL_BRANCH" \
  --vace-sample-steps 8 \
  --cuda-visible-devices <gpu>
```

The chain metadata reports `evidence_level: INTERFACE`, `visual_candidate: true`, and
`visual_quality_evaluated: false`. A comparison grid is for human review; it is not a
CONTROL/VISUAL/RESEARCH claim by itself.

## Acceptance checks to inspect

For every current run, inspect metadata for the canonical runtime contract:

```text
vace_runtime_inputs keys exactly:
  vace_conditioning_video
  quadmask_npy
  generation_mask
  operation
  vace_prompt
  frame_num

vace_conditioning_video.frame_0_is_edited_first_frame = true
vace_conditioning_video.future_frames_are_zero_filled = true
vace_conditioning_video.future_frames_source_video_used = false
generation_mask_values = [255]
generation_mask_is_full_domain = true
source_video_passed_to_vace = false
alignment_method present
```

A run that only produces files is INTERFACE evidence. Do not claim CONTROL, VISUAL,
or RESEARCH without the corresponding perturbation, review, or ablation evidence from
`docs/E2W_SPEC.md`.

## Static validation

Run the project tests after changing pipeline code or contracts:

```bash
$PY -m unittest discover -s tests -p 'test*.py'
```
