# E2W Spec

This is the single current specification for E2W. Superseded contracts, plans, and handoff notes are historical references only and live under `docs/archive/`.

## 1. Project Boundary

E2W studies source-conditioned counterfactual video editing with an explicit intervention contract:

```text
semantic intervention
+ first-frame-edited VACE conditioning video
+ explicit counterfactual description
+ grounded causal regions
+ operation-conditioned renderer
= counterfactual edited video
```

E2W is not a generic video editing benchmark, not pure text-to-video generation, and not a claim that the model already learns physical world models. The current goal is to make the counterfactual edit contract explicit and testable.

## 2. Planner Output Contract

The current compliant planner output schema is:

```text
e2w.planner_io.v6_executable.v1
```

The planner receives the original observed video, the user add/remove request, and any upstream context needed for provenance or audit. It must return one complete JSON object that can be validated and mapped into the current E2W runtime contract.

Required planner top-level keys:

```text
schema_version
video_id
task_type
edit_prompt
target_objects
protected_objects
event_summary
physical_causal_chain
counterfactual_expectation
quadmask_spec
quality_flags
```

Rules:

- `schema_version` must be `e2w.planner_io.v6_executable.v1`.
- `task_type` must be exactly `remove` or `add`.
- `task_type` must equal the downstream runtime `operation`.
- `counterfactual_expectation.if_removed` is the remove-side source for `vace_prompt` and must satisfy the remove prompt rules in this spec.
- `counterfactual_expectation.if_added` is the add-side edited-scene description source and must satisfy the add prompt rules in this spec.
- `quadmask_spec` must be executable enough to generate the authoritative `quadmask_npy`.
- The runner must store the planner/model JSON output as an upstream artifact.
- Runtime or adapter code must not silently rewrite planner JSON, replace `vace_prompt`, or substitute teacher/manual text to make a run pass.

### Executable Quadmask Spec

The nested `quadmask_spec` must use:

```text
schema_version = e2w.quadmask_spec.v1
```

Required executable fields:

```text
operation
primary.keyframes[].bbox_xyxy_norm1000
primary.keyframes[].positive_points_norm1000
affected.grid_shape
affected.frame_ranges[].cells
keep
```

Rules:

- `quadmask_spec.operation` must equal planner `task_type`.
- `primary.keyframes[].bbox_xyxy_norm1000` and `primary.keyframes[].positive_points_norm1000` use norm1000 image coordinates.
- `affected.grid_shape` and `affected.frame_ranges[].cells` define the affected non-target regions used to construct Q2.
- `keep` describes regions expected to remain unchanged and maps to Q3 semantics.
- Old empty forms such as `{"primary": {}, "affected": {}, "keep": {}}` are not executable and do not satisfy this spec.

The experimental schema `e2w.planner_output.v8_tool_augmented_grounding.v1` is not the current full-pipeline planner contract because it does not output executable `quadmask_spec` grounding.

## 3. Current Canonical Runtime Contract

There is one canonical runtime contract. Do not introduce parallel runtime contracts in new docs or reports.

### Required VACE runtime inputs

The VACE stage receives exactly these E2W-controlled inputs:

```text
vace_conditioning_video
quadmask_npy
generation_mask
operation
vace_prompt
frame_num
```

No `src_video`, `source_video`, `original_video`, or `factual_source_video` field is part of the VACE runtime input contract.

If an upstream stage needs to keep the original observed video for audit, provenance, or planner context, that is outside this VACE runtime contract. The VACE stage itself must consume only the first-frame-edited conditioning video as its visual condition.

## 4. VACE Conditioning Video

`vace_conditioning_video` is the only visual video condition passed to VACE.

It must be a first-frame-edited conditioning video:

```text
frame 0: edited first frame produced by the first-frame edit stage
future frames: conditioning-format frames prepared for VACE, not the original observed video
```

The original observed video must not be passed to VACE as `src_video` or any equivalent runtime input.

Rationale:

- VACE should be conditioned on the counterfactual first-frame state, not the factual original video.
- The runtime contract should not mix factual observation with renderer conditioning.
- Q3/keep semantics are carried by `quadmask_npy`, not by feeding original video through a separate source-video channel.

## 5. Quadmask

`quadmask_npy` is the semantic E2W control mask.

Required format:

```text
dtype: uint8
shape: [T, H, W]
allowed values: {0, 63, 127, 255}
```

Semantics:

| value | region | meaning |
|---:|---|---|
| `0` | Q0 | primary target or insertion region |
| `63` | Q1 | primary and affected overlap |
| `127` | Q2 | affected non-target region |
| `255` | Q3 | keep region |

Rules:

- `quadmask_npy` is authoritative.
- MP4 or image previews are not authoritative because compression can corrupt exact values.
- Resizing must use nearest-neighbor semantics.
- Shape/frame alignment must be explicit and recorded in metadata.
- Do not collapse Q0/Q1/Q2/Q3 into a binary semantic mask.

## 6. Generation Mask

`generation_mask` is a unified E2W-generated full-domain mask used only to satisfy the VACE known/generate interface.

Required semantics:

```text
all valid pixels in all VACE frames are generate-enabled
```

Required value convention:

```text
255 = generate
0   = invalid padding only, if a backend format absolutely requires padding
```

For normal E2W runs, `generation_mask` should be all `255` over the aligned `[T,H,W]` video domain.

Rules:

- `generation_mask` carries no E2W semantic edit meaning.
- Do not derive semantic meaning from `generation_mask`.
- Do not use backend-specific mask modes such as local edit masks or future-frame-only masks as E2W semantics.
- Do not use `generation_mask` to encode Q0/Q1/Q2/Q3.
- The only semantic region contract is `quadmask_npy`.

The required metadata should record:

```text
generation_mask_shape
generation_mask_values
generation_mask_is_full_domain: true
```

## 7. Operation

`operation` is a required structured control input.

Allowed values:

```text
remove
add
```

Rules:

- Do not rely only on natural-language prompt text to express the operation.
- `operation` must be passed as an explicit runtime input.
- Add/remove sensitivity must be evaluated separately from prompt wording.

## 8. VACE Prompt

The text input is named only:

```text
vace_prompt
```

Do not use parallel names such as `prompt`, `text_prompt`, `video_prompt`, `edited_scene_prompt`, or `counterfactual_prompt` in new specs/reports except when quoting historical artifacts.

`vace_prompt` describes the desired counterfactual edited scene.

Producer rule:

- `vace_prompt` must be produced by the actual upstream planner/model inference path for the run.
- Runtime/VACE adapter code must not silently invent, hard-code, rewrite, or replace `vace_prompt`.
- Experiment specs may define prompt constraints, but must not pre-fill the final `vace_prompt` string.
- The runner must store the planner/model output as an upstream artifact and pass it through unchanged to the VACE runtime.
- If planner output is weak, invalid, or visually poor, keep the run at the appropriate evidence level; do not substitute a teacher/manual prompt to make the pipeline look better.

Rules for remove:

- Must be target-free.
- Must not mention target names, aliases, visible target subparts, or target material terms.
- Must not use remove/delete/erase wording.
- Must not say `without <target>`, `no <target>`, `no longer present`, or `where <target> was`.

Rules for add:

- May mention the object to be added.
- Should describe the edited scene after addition.
- Must not contain removal-residue language such as absent/missing/gone/no-longer-visible for the added object.

Rules for both:

- The prompt should support the counterfactual scene, not replace structured controls.
- Region semantics belong to `quadmask_npy`.
- Operation semantics belong to `operation`.

## 9. Frame Count and Alignment

`frame_num` is required and must match the VACE-compatible temporal length.

Required metadata:

```text
frame_num
conditioning_video_shape
quadmask_shape
generation_mask_shape
alignment_required
alignment_method
```

Rules:

- `vace_conditioning_video`, `quadmask_npy`, and `generation_mask` must be aligned in frame count and spatial size before VACE execution.
- Silent frame-count or resolution mismatch is not allowed.
- Any alignment must be explicit and recorded.

## 10. Renderer Output

Required VACE output:

```text
edited_video
```

Required runtime metadata:

```text
vace_conditioning_video
quadmask_npy
generation_mask
operation
vace_prompt
frame_num
quadmask_values
generation_mask_values
generation_mask_is_full_domain
alignment_method
renderer_completed
edited_video_path
```

Output existence is not visual success. Non-black-frame checks are not visual success. Interface completion must be reported separately from control, visual, and research evidence.

## 11. Evidence Ladder

Every result must be reported with an evidence level:

```text
INTERFACE  = files/commands/metadata exist and run
STRUCTURAL = shapes/values/paths prove signals enter the intended model path
TRAINING   = loss/gate/gradient evidence from a real training run
CONTROL    = perturbation or swap proves response to operation/quadmask
VISUAL     = review confirms the edited video satisfies target criteria
RESEARCH   = ablation-backed evidence supports a paper-level claim
```

Do not upgrade evidence level without matching proof.

Examples:

- `edited_video` exists: INTERFACE.
- `quadmask_npy` has values `{0,63,127,255}` and reaches model-forward metadata: STRUCTURAL.
- add/remove swap changes output consistently: CONTROL.
- human review confirms correct object addition/removal, Q2 response, and Q3 preservation: VISUAL.

## 12. What This Spec Does Not Define

This spec intentionally does not define the training-data manifest or full training contract.

Do not infer training requirements from this runtime spec. Training data, cycle roles, side packets, target videos, and dataset audits will be specified separately when needed.

It also does not define checkpoint names, run directory names, or historical experiment schema names.

## 13. Current Do-Not-Do Rules

- Do not pass original/source/factual video to VACE as a runtime input.
- Do not use `src_video` as a field name in current VACE runtime docs.
- Do not use multiple names for the text input; use `vace_prompt` only.
- Do not use backend generation masks to encode E2W semantics.
- Do not introduce multiple generation-mask modes in current spec.
- Do not claim visual/control success from runtime completion.
- Do not revive archived docs as current constraints.
