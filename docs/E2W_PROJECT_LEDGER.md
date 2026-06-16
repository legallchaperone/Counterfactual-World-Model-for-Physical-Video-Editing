# E2W Project Ledger

> **Owner / update convention:** This file is the assistant-maintained continuity ledger for E2W. The user asked that Hermes maintains it. Treat it as a project state ledger, not a scratchpad. Update it only when a durable fact, boundary decision, verified artifact, or next-step priority changes. Do not use it for transient task notes.

Last updated: 2026-06-16T00:00:00Z
Maintainer skill: `e2w-project-ledger`
Canonical repo: `ssh cwx:/home/cwx/E2W`
Current spec: `/home/cwx/E2W/docs/E2W_SPEC.md`
Archived superseded docs: `/home/cwx/E2W/docs/archive/superseded-specs-20260608/`
Artifact root: `/data/cwx/E2W`
Python env: `/data/cwx/conda/envs/edit2world-phase1-real/bin/python`

---

## 1. North Star

E2W studies counterfactual video editing with an explicit intervention contract:

```text
semantic intervention
+ first-frame-edited VACE conditioning video
+ explicit counterfactual description
+ grounded causal regions
+ operation-conditioned renderer
= counterfactual edited video
```

Current canonical spec rule:

> The single current runtime/spec source is `docs/E2W_SPEC.md`. Superseded contracts, plans, and handoffs under `docs/archive/` are historical references only.

Reviewer one-liner:

> VOID's unit of supervision is a deleted video; E2W's unit of supervision is an intervention-conditioned counterfactual contract linking planner state, causal regions, operation control, and rendered video.

---

## 2. Current Planner and Runtime Boundary

The user clarified on 2026-06-09 that the current correct planner design should be called **Counterfactual Planner**, not a version-number route label. The existing schema id remains for artifact compatibility, but it is not the method or route name. Archived executable-planner materials are preserved under archive paths and must not be treated as current baselines.

Current Counterfactual Planner compatible schema id:

```text
e2w.planner_output.v8_tool_augmented_grounding.v1
```

Required planner fields:

```text
target_ref
edit_type
counterfactual_state
if_removed
```

The Counterfactual Planner is intentionally text/state-first. It does not directly output executable `quadmask_spec`; instead, the current intended pipeline is:

```text
original video + user remove request
-> Counterfactual Planner JSON
-> target_ref grounding with GroundingDINO/SAM2 or equivalent
-> quadmask_npy
-> vace_prompt from counterfactual_state / if_removed
-> first-frame-edited vace_conditioning_video
-> current six-input VACE runtime
```

The current VACE runtime contract is intentionally narrow.

Required VACE runtime inputs:

```text
vace_conditioning_video
quadmask_npy
generation_mask
operation
vace_prompt
frame_num
```

Durable boundary decisions:

- VACE runtime must not take `src_video`, `source_video`, `original_video`, or `factual_source_video` as inputs.
- The only visual condition passed to VACE is `vace_conditioning_video`.
- `vace_conditioning_video` must be a first-frame-edited conditioning video.
- `vace_conditioning_video` future frames must be zero/blank placeholders; they must not copy frames from the factual source video.
- `generation_mask` is unified full-domain generation, normally all `255` over the aligned video domain.
- `generation_mask` carries no semantic edit meaning.
- Region semantics live only in `quadmask_npy`.
- Text input is named only `vace_prompt`.
- Training manifests / target videos / side packets / cycle roles are not defined by the current runtime spec.
- Archived executable-planner schemas, tests, fixtures, tools, and artifacts are archived references only, not current planner baselines.

Unified interface decision (2026-06-16):

- New runs should use only `tools/e2w_remove.py`, `tools/e2w_add.py`, and `tools/e2w_add_then_remove.py`.
- Those three entry points share `tools/e2w_pipeline_core.py` and must emit the current six-input VACE runtime contract.
- Legacy names `tools/run_counterfactual_planner_pipeline.py`, `tools/run_add_pipeline_interface.py`, and `tools/run_add_then_remove_pipeline.py` are compatibility shims only; do not use them in new docs/reports/automation.
- Fixed run commands live in `docs/UNIFIED_PIPELINE_RUNBOOK.md`.
- Review fixes landed with the unification: remove schema/prompt/VACE failures fail loudly; add `primary_point` must lie inside `primary_bbox`; masked-inpaint metadata records backend mask consumption; VACE alignment metadata is recorded; chained add-then-remove uses evidence level `INTERFACE` plus `visual_candidate=true` rather than inventing a new evidence level.

---

## 3. Required Habit for Future E2W Conversations

Before answering any E2W status/design/progress question, load the `e2w-project-ledger` skill and read this file plus the current spec:

```bash
ssh cwx 'cd /home/cwx/E2W && sed -n "1,220p" docs/E2W_PROJECT_LEDGER.md && sed -n "1,260p" docs/E2W_SPEC.md'
```

If the question depends on current repo state, also run:

```bash
ssh cwx 'cd /home/cwx/E2W && git status --short --branch && git log -1 --oneline'
```

If the question depends on operational progress, read:

```bash
ssh cwx 'cd /home/cwx/E2W && sed -n "1,220p" STATUS.md'
```

Do not answer current project status purely from memory.

---

## 4. Current Repository Snapshot

Verified at: 2026-06-09T13:30:00Z after bridge structural gate and code-side contract fixes.

```text
repo: /home/cwx/E2W on ssh cwx
branch: main
head: d4dc4c6 修正v03生成掩码默认模式
tests: 17 tests OK
```

---

## 5. Workstream State

### A. Counterfactual Planner / Counterfactual State

Current understanding:

- Counterfactual Planner is the current correct planner design.
- Its compatible schema id is `e2w.planner_output.v8_tool_augmented_grounding.v1`; keep this string for artifact compatibility only.
- Counterfactual Planner is strong on parse/schema/target-free counterfactual text and should be the basis for current planner work.
- The Counterfactual Planner grounding bridge and runtime adapter have current-spec fixes for full-domain generation masks, E2W-level VACE input metadata, and adapter-name separation.
- Remove-side bridge STRUCTURAL gate passed on 30 sampled eval rows at `/data/cwx/E2W/runs/counterfactual_bridge_skipvace_30_20260609T_run`: planner parse/schema OK, GroundingDINO/SAM2 OK, all `quadmask_npy` value sets `[0,127,255]` with nonzero Q0/Q2, `generation_mask` values `[255]`, `vace_prompt_valid=true`, and `source_video_passed_to_vace=false`.
- One remove-side current-spec VACE INTERFACE smoke passed at `/data/cwx/E2W/runs/counterfactual_bridge_vace_interface_1_gpu2_20260609T_run` for sample `4fe6619a47`: first-frame edit OK, VACE backend returncode `0`, and output `/data/cwx/E2W/runs/counterfactual_bridge_vace_interface_1_gpu2_20260609T_run/edited_video_4fe6619a47.mp4` exists.
- Code-side bridge fix: current remove runner `tools/e2w_remove.py` (legacy `tools/run_counterfactual_planner_pipeline.py` is a shim) preserves SAM2 primary pixels as Q0 instead of collapsing them into Q1; Qwen Image Edit first-frame edit uses model CPU offload when available.
- Archived executable-planner materials are archived historical evidence.

Do not claim:

- Counterfactual Planner has CONTROL, VISUAL, or RESEARCH success from the current bridge/interface smoke;
- target-free text success alone implies renderer/control success;
- an archived executable-planner route is the current planner baseline.

### B. VACE / Quadmask Control

Current understanding:

- The current spec replaces older runtime variants with one VACE input contract.
- Runtime interface success does not prove learned quadmask semantics.
- The unmerged worktree `feat/phase1-v04-control-branch` is at `1f17ea4` and contains code-side training-format fixes aligned with `docs/E2W_SPEC.md`:
  - `edited_first_frame` conditioning replaces source-frame conditioning in `tools/train_v04_control_branch_real_overfit.py`;
  - full-domain generation mask is synthesized in the training script and recorded as non-semantic;
  - `vace_prompt` naming is required by the training script;
  - Q3 latent MSE loss is present.
- The historical real 14B overfit run `/data/cwx/E2W/checkpoints/v04_real_overfit_14b_20260604` predates those fixes. It reached `final_gate = 0.022334493696689606`, but its `metrics.jsonl` does not contain the new Q3/full-domain-mask metadata fields. Treat it as stale training evidence.
- The corrected real 14B self-insertion overfit completed at `/data/cwx/E2W/checkpoints/v04_real_overfit_14b_specfix_selfinsert_20260612`, after a 20-step pilot at `/data/cwx/E2W/checkpoints/v04_real_overfit_14b_specfix_selfinsert_20260612_pilot20`. The run used real weights, 200 steps, 201 metrics rows, current-spec conditioning inputs `["edited_first_frame","quadmask","vace_prompt"]`, full-domain generation mask values `[255]`, Q3 loss weight `0.1`, nonzero gradients, and final gate `0.046739354729652405`.
- The current branch tests verified during audit: `tests.test_v04_anchor_manifest_audit`, `tests.test_v04_control_branch_freeze`, and `tests.test_v04_control_branch_gradients` ran 19 tests OK.
- Evidence level is TRAINING only. Real evidence still required: connect the trained v04 branch checkpoint to VACE inference, then run operation swap, quadmask perturbation, Q2 response, Q3 preservation, and ablations.
- On 2026-06-13, the first Physics-IQ simple-eval run under `/data/cwx/E2W/runs/physics_iq_for_simple_eval` was invalidated for remove-side visual judging: `run_counterfactual_planner_pipeline.py` wrote `vace_conditioning_video` with frame 0 edited but future frames copied from the factual source video. This explains the observed target reappearance after the first frame and violates the current conditioning-video contract. The runner was fixed to write edited frame 0 plus zero-filled future frames, and `physics_iq_for_simple_eval validate-run` now rejects runs whose conditioning future frames use source video.

Do not claim:

- VACE has learned add/remove semantics from interface success;
- control branch works visually without sensitivity tests;
- generation-mask behavior proves E2W semantic control;
- the stale 14B run proves the corrected training format works.

### C. Data / Anchors

Current understanding:

- Planner seed data exists for text experiments; it is not full paired physics supervision.
- Phase1A Pexels self-insertion data exists and is add/remove balanced.

Verified add/remove self-insertion anchor data:

```text
/data/cwx/E2W/data/phase1a_pexels_self_insert_v1/03_self_insert/manifests/self_insert_train.jsonl: 16 rows = 8 add + 8 remove
/data/cwx/E2W/data/phase1a_pexels_self_insert_v1/03_self_insert/manifests/overfit_16.jsonl: 16 rows = 8 add + 8 remove
/data/cwx/E2W/data/phase1a_pexels_self_insert_v1/03_self_insert/manifests/eval_4.jsonl: 8 rows = 4 add + 4 remove
```

Do not claim:

- self-insertion is full physical dataset;
- Kubric/HUMOTO coverage is complete;
- Phase1A proves physics.

### D. Add Pipeline

Current understanding:

- Add operation is contract-supported in schema, prompt serialization, VACE prompt validation, unified runner wrappers, and tests.
- Current add entry point is `tools/e2w_add.py`; current add-then-remove entry point is `tools/e2w_add_then_remove.py`; both share `tools/e2w_pipeline_core.py`.
- The add path now uses the current first-class add design: add planner output (`target_ref`, `vace_prompt`, `primary_point`, optional `primary_bbox`) -> planner-region masked inpaint first-frame edit -> SAM2 on the edited first frame -> change-overlap consistency guard -> `quadmask_npy` -> VACE(add).
- `tools/run_add_pipeline_interface.py` is now only a deprecated compatibility shim to `tools/e2w_add.py`; new runs/docs should not use it.
- Verified add INTERFACE run after the masked-inpaint update exists in the branch status (`e2w_add` smoke with overlap 0.796 and `edited_video` output). Evidence level remains INTERFACE only; visual quality was not evaluated.
- Historical add run `/data/cwx/E2W/runs/add_pipeline_interface_add_bg_000001_20260609T024340Z` had a remove-residue prompt gap and remains only provenance/interface history, not contract-safe add prompt evidence.
- Contract-safe add v2 run `/data/cwx/E2W/runs/add_pipeline_interface_add_bg_000001_v2_20260609T152335Z` fixed the prompt path but predates the final unified masked-inpaint interface.
- Add visual success, add/remove operation control, and learned VACE add semantics are not established.

Important historical run:

```text
/data/cwx/E2W/runs/e2w_v0_3_full_add_0076_pipeline_20260602T121720Z
```

Claim boundary:

> The add/add-then-remove interfaces are unified at current-spec INTERFACE level. This proves the code path and metadata contract, not learned planner generalization, renderer control, visual success, or research-level counterfactual editing.

---

## 6. Evidence Standard

Every future claim should be classified as one of:

```text
INTERFACE: command/contract/path exists and runs.
STRUCTURAL: tests or shapes/values confirm signal enters model path.
TRAINING: loss/gate/gradient evidence from real training run.
CONTROL: perturbation or swap shows model responds to operation/quadmask.
VISUAL: human/model review confirms edited video satisfies target criteria.
RESEARCH: ablation-backed evidence supports paper-level claim.
```

Interface success alone must never be reported as visual/control/research success.

---

## 7. Minimal Viable Next Proof

The bridge structural spine is now proven:

```text
Completed (2026-06-09):
  - planner JSON parse/schema: 30/30 eval rows OK
  - target_ref -> GroundingDINO/SAM2 grounding: OK
  - quadmask_npy values [0,127,255], nonzero Q0/Q2: OK
  - generation_mask full-domain all-255: OK
  - vace_prompt target-free and planner-produced: OK
  - metadata links planner JSON -> grounding -> quadmask -> VACE inputs: OK
  - remove-side VACE INTERFACE smoke: 1 sample, returncode 0
```

The next meaningful E2W proof is CONTROL evidence:

```text
Design and run operation swap + quadmask perturbation tests:
  - swap operation add <-> remove on same clip, same quadmask: outputs must differ;
  - zero out Q0/Q2 region (all-255 quadmask): object should not be removed;
  - swap Q0 location to a non-target region: edit region should shift;
  - Q3 pixels in reference frame should be preserved across the edit.
Only after CONTROL evidence passes is VISUAL review meaningful.
```

---

## 8. Do-Not-Do List

1. Do not pass original/source/factual video to VACE as a runtime input.
2. Do not use `src_video` as a current VACE runtime field name.
3. Do not use multiple prompt names; use `vace_prompt` only.
4. Do not use generation-mask modes to encode E2W semantics.
5. Do not weaken tests, schema, validators, prompt contracts, or target-free rules to pass a run.
6. Do not silently rewrite target-contaminated prompts.
7. Do not treat `edited_video` existence, non-black frames, or signal consumption metadata as visual success.
8. Do not claim learned add pipeline from the 0076 add-mug teacher/manual run.
9. Do not revive archived docs as current constraints.
10. Do not use archived executable-planner artifacts as current baselines.

---

## 9. Update Policy

Update this file when:

- a durable boundary/spec decision changes;
- a new canonical run/checkpoint/dataset becomes the reference;
- a workstream moves from interface → structural → training → control → visual evidence;
- a next step is completed or superseded;
- a user explicitly corrects project framing.

Do not update for temporary commands, speculative ideas, raw logs, or per-turn progress notes.
