# E2W Runtime Contract

Last updated: 2026-06-03 UTC

This document defines the runtime contract for the local E2W smoke pipeline. It separates historical interface versions from the current v0.3 target so that reports do not overstate what the model actually consumed.

The canonical planner I/O schema is `e2w.planner_io.v6_executable.v1`.

## Version Summary

| version | status | VACE input contract | quadmask status | main purpose |
|---|---|---|---|---|
| `v0` | legacy integration | Wan-VACE consumes `src_video` plus binary `src_mask` | saved or logged only | prove the external VACE executor can run |
| `v0.2` | frozen smoke pipeline | Wan-VACE consumes Qwen first-frame conditioning plus known/generate mask | built, packaged, and audited, but not consumed by VACE | SFT VLM planner inference, fresh artifacts, strict prompt/report contracts |
| `v0.3` | current quadmask-control contract | E2W quad VACE runner consumes `src_video`, binary generation mask, `quadmask.npy`, and `operation` | model control path consumes four-value quadmask | prove the VACE interface can accept semantic E2W quadmask control |

## Planner I/O Schema

New planner train/eval/smoke inputs must use
`e2w.planner_io.v6_executable.v1`. This schema supports both `remove` and
`add`, but operation support is not enough by itself. The planner must also
produce executable grounding:

- top-level `video_id`, `task_type`, `edit_prompt`, `target_objects`,
  `protected_objects`, `event_summary`, `physical_causal_chain`,
  `counterfactual_expectation`, `quadmask_spec`, and `quality_flags`
- `quadmask_spec.schema_version: e2w.quadmask_spec.v1`
- `quadmask_spec.operation: remove|add`
- `quadmask_spec.primary.keyframes[].bbox_xyxy_norm1000`
- `quadmask_spec.primary.keyframes[].positive_points_norm1000`
- `quadmask_spec.affected.grid_shape`
- `quadmask_spec.affected.frame_ranges[].cells`

The old text-only `quadmask_spec.primary/affected/keep` prompt is invalid for
new data. It may only remain in archived source files. Parser behavior is
strict: partial text or nested JSON objects are failures, even if an internal
object is valid JSON.

The executable schema checks are enforced by `tests/test_v02_contracts.py`.
Do not weaken those checks to pass a run; fix the planner data, prompt, or
checkpoint.

## Planner User Prompt Contract

Canonical planner train/eval/smoke JSONL rows use the prompt emitted by
`tools/e2w_v0_common.py::build_planner_user_prompt`.

The prompt layout is fixed:

1. planner role and `e2w.planner_io.v6_executable.v1` schema version
2. task operation
3. full planner schema JSON
4. current task with exact `video_id` and user request
5. final rules for the current task

Do not add one-shot wrapper objects such as `input_video_id`,
`input_user_request`, `good_complete_output`, or `output_excerpt`. Prompt-only
experiments showed the current LoRA copies those wrapper keys and fails strict
top-level planner JSON parsing. If examples are needed, they must be trained as
assistant labels, not embedded as wrapper examples in the inference prompt.

The final rules are part of the schema contract:

- output exactly one complete top-level planner JSON object
- include executable quadmask grounding fields
- use norm1000 coordinates for bbox and points
- use A1-style grid cells with A1 at the top-left
- never emit the archived empty `quadmask_spec` schema
- set `quadmask_spec.operation` equal to `task_type`
- for remove operations, `counterfactual_expectation.if_removed` is copied
  into the VACE video prompt and must be target-free
- forbidden in remove `if_removed`: the removed target name, aliases, visible
  target parts, target material words, and negative wording such as
  `no <target>`, `without <target>`, `<target> is removed`, or
  `<target> is no longer present`
- remove `if_removed` should describe only non-target objects, local
  background, revealed surfaces, lighting/shadow/reflection changes, and
  resulting physical motion

Train and eval JSONLs must use the same prompt contract. Rewriting only eval
prompts is an invalid comparison because the SFT LoRA is sensitive to prompt
distribution shift.

## v0.2 Runtime Contract

The canonical v0.2 runner is:

```bash
tools/run_v02_qwen_vace_smoke.py
```

Stage order:

1. SFT VLM planner inference
2. quadmask build
3. Qwen first-frame edit
4. legacy VACE v0 generation
5. package/report
6. artifact freshness audit

The planner stage is valid only when `tools/eval_vlm_planner.py` runs the SFT
Qwen2.5-VL LoRA planner and writes both `raw_output.txt` and `raw.pred.json`
for every selected sample. A run without those raw prediction artifacts is not
a valid full forward pass.

v0.2 VACE consumes:

- `vace_conditioning.mp4`: edited first frame followed by black frames
- `vace_generation_mask.mp4`: frame 0 known, future frames generated
- neutral target-free VACE prompt

v0.2 VACE does not consume:

- `quadmask.npy`
- `quadmask.mp4`
- formal `0/63/127/255` E2W edit semantics

Therefore v0.2 reports may say `quadmask_valid` or `quadmask_saved`, but must not say `quadmask_consumed_by_backend: true`.

## v0.3 Runtime Contract

v0.3 upgrades the VACE stage contract only. It does not imply that the quadmask branch has been trained.

A v0.3 VACE run must pass these interface gates:

- VACE command receives `--quadmask_npy`.
- VACE command receives `--operation remove|add`.
- VACE command receives a binary `--generation_mask`.
- The E2W quad runner expands Wan-VACE context from legacy `96` channels to `416` channels.
- The extra quadmask channels are part of the VACE control/context branch used during model forward.
- Output metadata records `quadmask_passed_to_backend_command: true` when the command contains `--quadmask_npy`, `--operation`, and `--generation_mask`.
- Output metadata records `quadmask_consumed_by_backend: true` only when VACE was actually run, returned `0`, and produced `edited_video.mp4`.

The canonical direct v0.3 experiment wrapper is:

```bash
tools/run_vace_v03_quad_experiment.py
```

It calls:

```bash
tools/run_wan_vace_quad_i2v.py
```

which imports:

```bash
tools/e2w_vace_quad_i2v.py
```

## Quadmask Semantics

`quadmask.npy` is a `uint8` array with shape `[T,H,W]`.

Allowed values:

| value | meaning |
|---:|---|
| `0` | primary target region |
| `63` | primary and affected overlap |
| `127` | affected non-target region |
| `255` | keep region |

v0.3 must preserve these values. If resizing or frame-count alignment is needed, it must be explicit and recorded in metadata. Silent 80/81 frame mismatch is not allowed.

## Binary Generation Mask

Wan-VACE still requires a binary generation mask:

| value | meaning |
|---:|---|
| `0` | keep/known source pixels |
| `255` | generate pixels |

In v0.3 this binary mask is not the semantic edit mask. It is only the known/generate gate. The semantic edit mask is `quadmask.npy`.

Supported v0.3 generation-mask modes:

- `quadmask-editable`: generate where `quadmask != 255`; keep elsewhere.
- `future-full-frame`: keep frame 0, generate future full frames. This matches v0.2 behavior but still feeds quadmask through the VACE control path.

## Current Training Status

v0.3 means "VACE can consume quadmask" at the interface/model-forward level. It does not mean the added quadmask channels have learned useful semantics.

Current quad runner initialization:

- old Wan-VACE 96-channel patch embedding weights are copied
- new quadmask channels are zero-initialized
- add/remove operation embedding is zero-initialized

This preserves checkpoint behavior before training. Fine-tuning or adapter training is still required before expecting reliable semantic control.

## Reporting Rules

Reports must distinguish:

- `vace_completed`
- `quadmask_available`
- `quadmask_frame_aligned`
- `quadmask_passed_to_backend_command`
- `quadmask_consumed_by_backend`
- `quadmask_trained_or_finetuned`
- visual success

Interface success is not visual success. A v0.3 run with zero-initialized quad channels can prove that the interface and model-forward path work, but cannot prove that VACE visually obeyed the quadmask.

## v0.3 Add-Mug Experiment Template

```bash
cd /home/cwx/E2W
RUN_DIR=/data/cwx/E2W/runs/e2w_v0_3_quad_vace_add_0076_$(date -u +%Y%m%dT%H%M%SZ)
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/run_vace_v03_quad_experiment.py \
  --src-video /home/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z/0076/edited_video.mp4 \
  --prompt "Add a yellow mug on the rotating turntable in front of the spotlight" \
  --quadmask-npy /home/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z/0076/quadmask.npy \
  --operation add \
  --run-dir "$RUN_DIR" \
  --generation-mask-mode quadmask-editable \
  --align-quadmask nearest \
  --cuda-visible-devices 4 \
  --sample-steps 8 \
  --run-vace
```

For this sample, `--align-quadmask nearest` is required because the source v0.2 quadmask is 80 frames while the VACE source video is 81 frames.
