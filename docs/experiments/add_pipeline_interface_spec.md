# Add Pipeline INTERFACE Experiment Spec

Status: draft for user review  
Branch/worktree: `feat/add-pipeline` at `/home/cwx/E2W/.worktree/feat/add-pipeline`  
Scope: experiment implementation only; do not change `main` until PR review

---

## 1. Purpose

This experiment tries to run the **add** direction of E2W at **INTERFACE** evidence level.

The short-term goal is not to prove visual quality, physical reasoning, learned planner add ability, or learned VACE quadmask semantics. The goal is only to verify that an add request can pass through the current runtime contract and produce a VACE edited-video artifact with correctly shaped and valued add-side masks.

Success means:

```text
operation = add
Qwen Edit produces edited_first_frame
quadmask_npy is built from edited_first_frame using SAM2
quadmask_npy has shape [T,H,W]
quadmask_npy values are a subset of {0,63,127,255}
generation_mask is full-domain all 255
VACE receives the current six-input runtime contract
VACE produces edited_video path
metadata records all above facts
```

Evidence level if successful: **INTERFACE**.

---

## 2. Non-Goals

Do not claim or optimize for:

- VACE visual success;
- object permanence quality;
- physical plausibility;
- learned planner add ability;
- learned VACE add semantics;
- Q2 physical consequence correctness;
- comparison to remove, VOID, or any baseline;
- training-data manifest design.

This is a smoke/interface experiment, not a research result.

---

## 3. Current Runtime Contract

The experiment must follow `docs/E2W_SPEC.md`.

Required VACE runtime inputs are exactly:

```text
vace_conditioning_video
quadmask_npy
generation_mask
operation
vace_prompt
frame_num
```

Hard constraints:

- Do not pass original/source/factual video to VACE as `src_video`, `source_video`, `original_video`, or equivalent E2W runtime input.
- The only visual condition passed to VACE is `vace_conditioning_video`.
- `vace_conditioning_video` is built from the Qwen-Edit `edited_first_frame`.
- `generation_mask` is full-domain all-generate, normally all `255` over `[T,H,W]`.
- `generation_mask` carries no Q0/Q1/Q2/Q3 semantics.
- Region semantics live only in `quadmask_npy`.
- Text input name is `vace_prompt`.
- `operation` must be explicit and equal to `add`.

If the current legacy VACE wrapper still uses backend argument names such as `--src_video` or `--prompt`, the implementation may use those only inside an adapter layer, but experiment metadata must record the E2W-level contract names above and must not expose legacy names as the current E2W contract.

---

## 4. Data Source

This experiment must use an existing, local, low-risk source video. It must not depend on downloading new data.

Primary data source:

```text
/data/cwx/E2W/data/phase1a_pexels_self_insert_v1/02_background_clean/videos_mp4/
```

These are already-filtered clean-background Pexels videos from Phase1A. They are suitable for an add INTERFACE smoke because the scene is simple and adding a small object should not require complex physical interpretation.

Default sample for the first run:

```text
source_video_for_upstream_context: /data/cwx/E2W/data/phase1a_pexels_self_insert_v1/02_background_clean/videos_mp4/bg_000001.mp4
target_ref: red mug
add_instruction: Add a red mug on the table near the center of the image.
planner_prompt_requirement: actual upstream planner/model inference must produce `vace_prompt`; no teacher/manual prompt fallback
frame_num: 21 for first smoke; 81 optional after the small run passes
```

Important distinction:

- The clean-background video is allowed as **upstream input** to extract the original first frame for Qwen Edit.
- The clean-background/original video must not be passed to VACE as a source/original video.
- VACE receives only `vace_conditioning_video`, whose first frame is the Qwen-Edit `edited_first_frame`.

Do not use the existing Phase1A self-insertion `target_video` or precomputed `edited_first_frame` as the main success path, because this experiment specifically tests the Qwen Edit add interface. Existing Phase1A add records may be used only as debugging references or fallback diagnostics, and fallback usage must not be reported as the primary acceptance run.

If `bg_000001.mp4` is unsuitable after first-frame inspection, choose another video from the same clean-background directory and record the chosen path and reason in `metadata.json`.

## 5. VACE Prompt Source

`vace_prompt` must come from the actual upstream planner/model inference path used by this pipeline run.

Correct source rule:

```text
original video + user prompt
        │
        ▼
planner/model inference
        │
        ▼
vace_prompt artifact
        │
        ▼
VACE runtime input
```

The runner is responsible for orchestrating the real pipeline, not for authoring `vace_prompt` itself. It must consume the planner/model output and pass it through unchanged.

Hard constraints:

- Do not pre-fill the final `vace_prompt` in this experiment spec.
- Do not silently hard-code, invent, rewrite, or replace `vace_prompt` in runner code.
- Do not use teacher/manual prompts for the acceptance run.
- Do not claim learned planner add quality from this INTERFACE smoke.
- If the planner output is weak but schema-valid enough to run, run it and record that limitation.
- For add INTERFACE quadmask construction, planner/model output with a valid primary point but missing bbox is acceptable, because SAM2 can be prompted by point on `edited_first_frame`; record `accepted_point_only_for_add_interface=true` in metadata.
- If the planner output is invalid for the pipeline contract, the runner may retry planner/model inference up to 3 total attempts.
- Retry attempts must use the same original video and user prompt, optionally with a stricter system/instruction reminder to satisfy the schema; do not manually edit the model output.
- If all 3 attempts are invalid, fail loudly and record the blocker; do not patch around it with a manual prompt.

For add, the planner-produced `vace_prompt` may mention the added object. It should describe the edited scene, not issue a command.

Required metadata:

```json
{
  "vace_prompt_source": "planner_model",
  "vace_prompt_passed_through_unchanged": true,
  "planner_attempt_count": 1,
  "planner_invalid_attempt_errors": [],
  "planner_output_manually_modified": false,
  "manual_or_teacher_vace_prompt_used": false,
  "learned_planner_add_quality_claimed": false
}
```

## 6. Proposed Add Pipeline
## 6. Proposed Add Pipeline

Input:

```text
original_video
add instruction / target_ref
optional placement text
frame_num
```

Pipeline:

```text
original first frame
+ add instruction / target_ref / placement
        │
        ▼
Qwen Edit
        │
        ▼
edited_first_frame
        │
        ├── build vace_conditioning_video
        │       frame 0 = edited_first_frame
        │       future frames = VACE conditioning-format frames, not original video
        │
        └── SAM2 on edited_first_frame with add target prompt/points/box
                │
                ▼
             primary added-object mask
                │
                ▼
             quadmask_npy
                Q0 = added object region
                Q2 = local affected band / image-diff affected region
                Q3 = keep region
                Q1 = optional overlap/contact region if constructed

VACE runtime receives:
  vace_conditioning_video
  quadmask_npy
  generation_mask = all 255
  operation = add
  vace_prompt
  frame_num
        │
        ▼
edited_video
```

---

## 7. Qwen Edit Role

Qwen Edit is used only as a **first-frame counterfactual materializer**.

It is responsible for:

```text
original first frame + add instruction -> edited_first_frame
```

It is not evidence that:

- the planner learned add;
- the renderer learned add;
- the system understands physics.

---

## 8. Quadmask Construction for Add

Because the added object is not present in the original first frame, add-side quadmask is constructed from `edited_first_frame`.

### 8.1 Primary mask

Use SAM2 on `edited_first_frame` to segment the added target.

Accepted prompt sources, in priority order:

1. explicit user/experiment-provided point or box on the edited frame;
2. GroundingDINO/text grounding on `edited_first_frame` using `target_ref`, then SAM2;
4. simple diff-assisted fallback only if the implementation records fallback usage clearly.

The primary added-object mask becomes:

```text
Q0 = 0
```

### 8.2 Affected region

For this INTERFACE experiment, Q2 does not need to be physically perfect. It must be deterministic and recorded.

Recommended Q2 heuristic:

```text
image_diff = abs(edited_first_frame - original_first_frame)
Q2 candidates =
  dilate(Q0) outside Q0
  union diff-threshold pixels outside Q0
  union small contact/support band near bottom of Q0, if applicable
```

Then:

```text
Q2 = 127
Q3 = 255 everywhere else
```

Q1 may be omitted or set only for explicitly constructed overlap/contact pixels:

```text
Q1 = 63
```

For the smoke criterion, it is acceptable for `quadmask_npy` to contain only:

```text
{0,127,255}
```

as long as this subset is recorded and valid.

### 8.3 Temporal propagation

For INTERFACE level, the simplest valid temporal strategy is:

```text
repeat first-frame quadmask for all T frames
```

This is not a visual/control claim. It is only an interface smoke strategy.

Optional improvement:

```text
SAM2 propagate primary mask through vace_conditioning_video if available and reliable
```

But the minimum acceptance criterion is repeated first-frame quadmask with correct shape/value audit.

---

## 9. VACE Prompt for Add

The field must be named:

```text
vace_prompt
```

The experiment spec does not predefine the final `vace_prompt`. It defines only constraints and provenance.

For add, the upstream planner/model-produced `vace_prompt` may mention the added object. It should describe the edited scene, not issue a command.

Good form expected from planner/model:

```text
A ceramic mug sits on the wooden table near the center, with a soft contact shadow and lighting consistent with the room.
```

Invalid runner behavior:

```text
# hard-coded by runner/spec instead of planner/model
A ceramic mug sits on the wooden table near the center...
```

Avoid prompt content such as:

```text
Add a mug to the video.
The mug is no longer absent.
A missing mug appears.
```

Required metadata:

```json
{
  "vace_prompt_source": "planner_model",
  "vace_prompt_passed_through_unchanged": true,
  "planner_attempt_count": 1,
  "planner_invalid_attempt_errors": [],
  "planner_output_manually_modified": false,
  "manual_or_teacher_vace_prompt_used": false,
  "learned_planner_add_quality_claimed": false
}
```

---

## 10. Required Output Layout

Each run should write under:

```text
/data/cwx/E2W/runs/add_pipeline_interface_<sample_id>_<timestamp>/
```

Required files:

```text
input_first_frame.png
edited_first_frame.png
vace_conditioning_video.mp4
quadmask.npy
generation_mask.npy or VACE-compatible generation mask file
quadmask_preview.png or .mp4
vace_prompt.txt
edited_video.mp4
metadata.json
run_command.json
```

If the backend produces a different edited-video name, symlink or copy it to:

```text
edited_video.mp4
```

---

## 11. Required Metadata

`metadata.json` must contain at least:

```json
{
  "evidence_level": "INTERFACE",
  "operation": "add",
  "target_ref": "...",
  "vace_prompt": "...",
  "vace_prompt_source": "planner_model",
  "vace_prompt_passed_through_unchanged": true,
  "planner_attempt_count": 1,
  "planner_invalid_attempt_errors": [],
  "planner_output_manually_modified": false,
  "manual_or_teacher_vace_prompt_used": false,
  "learned_planner_add_quality_claimed": false,
  "frame_num": 21,
  "vace_runtime_inputs": {
    "vace_conditioning_video": "...",
    "quadmask_npy": "...",
    "generation_mask": "...",
    "operation": "add",
    "vace_prompt": "...",
    "frame_num": 21
  },
  "source_video_passed_to_vace": false,
  "legacy_backend_arg_adapter_used": true,
  "qwen_edit": {
    "backend": "...",
    "edited_first_frame": "..."
  },
  "sam2": {
    "used_for_quadmask": true,
    "prompt_source": "point|box|grounding_dino|fallback",
    "primary_mask_shape": [480, 832],
    "primary_mask_area": 12345
  },
  "quadmask": {
    "shape": [21, 480, 832],
    "dtype": "uint8",
    "values": [0, 127, 255],
    "q0_area_mean": 0.0,
    "q2_area_mean": 0.0,
    "q3_area_mean": 0.0,
    "temporal_strategy": "repeat_first_frame"
  },
  "generation_mask": {
    "shape": [21, 480, 832],
    "dtype": "uint8",
    "values": [255],
    "generation_mask_is_full_domain": true,
    "generation_mask_encodes_quadmask_semantics": false
  },
  "edited_video": {
    "path": ".../edited_video.mp4",
    "exists": true,
    "size_bytes": 0
  },
  "success_criteria": {
    "edited_first_frame_exists": true,
    "quadmask_shape_ok": true,
    "quadmask_values_ok": true,
    "generation_mask_full_domain": true,
    "edited_video_exists": true
  }
}
```

Do not record visual success unless a separate visual review is performed. For this experiment, metadata should explicitly say visual success is not evaluated.

---

## 12. Acceptance Criteria

The experiment is successful only if all are true:

1. New branch/worktree is `feat/add-pipeline`; no `main` changes are required.
2. The upstream source video comes from the clean-background Phase1A directory, with the chosen path recorded in metadata.
3. There is a run directory under `/data/cwx/E2W/runs/`.
4. Planner/model inference was attempted on the original video + user prompt, with at most 3 attempts if outputs are invalid.
5. No planner output was manually edited, manually completed, or replaced by a hand-written output.
6. `edited_first_frame.png` exists and was produced by Qwen Edit or its configured image-edit backend.
7. `quadmask.npy` was generated from `edited_first_frame` using SAM2 in the main path.
8. `quadmask.npy` has shape `[frame_num,H,W]`.
9. `quadmask.npy` dtype is `uint8`.
10. `quadmask.npy` values are a subset of `{0,63,127,255}`.
11. `generation_mask` is all `255` over the same `[frame_num,H,W]` domain.
12. VACE is invoked with the E2W-level six-input contract recorded in metadata.
13. `edited_video.mp4` exists and is non-empty.
14. `metadata.json` records that evidence level is `INTERFACE` and visual quality is not evaluated.
15. `metadata.json` records `vace_prompt_source=planner_model`, `vace_prompt_passed_through_unchanged=true`, `manual_or_teacher_vace_prompt_used=false`, and `learned_planner_add_quality_claimed=false`.
16. `metadata.json` records planner retry count and all invalid-attempt errors, if any.
17. Relevant smoke tests/audit script pass.

---

## 13. Expected Code Changes

Likely additions on `feat/add-pipeline`:

```text
tools/run_add_pipeline_interface.py
tools/build_add_quadmask_from_edited_first_frame.py
tests/test_add_pipeline_interface.py
docs/experiments/add_pipeline_interface_spec.md
```

The implementation should reuse existing utilities where safe, but it must not weaken existing validators or rewrite `docs/E2W_SPEC.md`.

---

## 14. pi Agent Assignment After Approval

After user approval, instruct the `pi` agent on `ssh cwx` to implement only this spec on the new worktree:

```text
/home/cwx/E2W/.worktree/feat/add-pipeline
```

Rules for the agent:

- Do not modify `/home/cwx/E2W` main checkout.
- Do not change `docs/E2W_SPEC.md` unless explicitly asked.
- Do not weaken tests/contracts.
- Build a minimal runnable add interface experiment.
- Prefer one small sample and short frame count first.
- Stop and report blockers if Qwen Edit, SAM2, or VACE credentials/checkpoints are unavailable.
- Return exact commands, run directory, metadata path, and test output.

---

## 15. Review / PR Plan

After implementation:

1. Hermes reviews code diff and tests.
2. Hermes inspects run artifacts and `metadata.json`.
4. Hermes confirms no `main` worktree changes.
5. Hermes commits the branch if not already committed.
6. Hermes opens a PR from `feat/add-pipeline` to `main`.
7. User reviews PR before merge.
