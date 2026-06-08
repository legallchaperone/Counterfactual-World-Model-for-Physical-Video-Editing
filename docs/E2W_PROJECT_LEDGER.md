# E2W Project Ledger

> **Owner / update convention:** This file is the assistant-maintained continuity ledger for E2W. The user asked that Hermes maintains it. Treat it as a project state ledger, not a scratchpad. Update it only when a durable fact, boundary decision, verified artifact, or next-step priority changes. Do not use it for transient task notes.

Last updated: 2026-06-08T16:10:14Z
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

## 2. Current Runtime Boundary

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
- `generation_mask` is unified full-domain generation, normally all `255` over the aligned video domain.
- `generation_mask` carries no semantic edit meaning.
- Region semantics live only in `quadmask_npy`.
- Text input is named only `vace_prompt`.
- Training manifests / target videos / side packets / cycle roles are not defined by the current runtime spec.

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

Verified at: 2026-06-08T16:10:14Z before the spec cleanup commit.

```text
repo: /home/cwx/E2W on ssh cwx
branch: main
status before cleanup: main...origin/main [ahead 6]
head before cleanup: 918ca7c A13 加入 first frame edit，完善端到端 pipeline
```

---

## 5. Workstream State

### A. Planner / Counterfactual State

Current understanding:

- The executable planner route supports full executable schema but strict remove gates remain low.
- The planner-text route is strong on parse/schema/target-free counterfactual text but does not output executable quadmask grounding.
- Text-planner success should not be treated as downstream renderer success.

Do not claim:

- current planner can run full forward pass cleanly;
- target-free text success implies renderer/control success.

### B. VACE / Quadmask Control

Current understanding:

- The current spec replaces older runtime variants with one VACE input contract.
- Runtime interface success does not prove learned quadmask semantics.
- Real evidence required: operation swap, quadmask perturbation, Q2 response, Q3 preservation, and ablations.

Do not claim:

- VACE has learned add/remove semantics from interface success;
- control branch works visually without sensitivity tests;
- generation-mask behavior proves E2W semantic control.

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

- Add operation is contract-supported in schema, prompt serialization, VACE prompt validation, runner wrappers, and tests.
- A single 0076 add-mug full pipeline artifact exists and passed machine/interface checks under older docs.
- That 0076 run used teacher/manual artifacts, not learned VLM planner add inference.
- Add masks in the 0076 run are executable but coarse.
- Add visual success is not established.

Important historical run:

```text
/data/cwx/E2W/runs/e2w_v0_3_full_add_0076_pipeline_20260602T121720Z
```

Claim boundary:

> Add pipeline is contract-supported and has one teacher/manual smoke success; learned planner add and learned VACE add semantics are not established.

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

The next meaningful E2W proof should be small and controlled:

```text
Use paired/self-insertion anchors.
Train only the relevant E2W control path.
Evaluate:
  - overfit loss decreases;
  - Q3 preservation remains high;
  - operation swap add↔remove changes output/context appropriately;
  - Q0 perturbation changes primary edit;
  - Q2 perturbation changes affected consequences;
  - binary-only / no-counterfactual-state ablations are worse.
```

Avoid long full-pipeline runs until this spine is proven.

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

---

## 9. Update Policy

Update this file when:

- a durable boundary/spec decision changes;
- a new canonical run/checkpoint/dataset becomes the reference;
- a workstream moves from interface → structural → training → control → visual evidence;
- a next step is completed or superseded;
- a user explicitly corrects project framing.

Do not update for temporary commands, speculative ideas, raw logs, or per-turn progress notes.
