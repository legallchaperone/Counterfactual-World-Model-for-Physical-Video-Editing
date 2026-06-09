# E2W Agent Instructions

This file is the canonical operating manual for coding/research agents working in this repository.

E2W is a research project. Correctness is not just whether code runs; correctness is whether a claim is backed by the right level of evidence.

## 0. Read First

Before any non-trivial E2W work, read these in order:

```bash
sed -n '1,260p' docs/E2W_SPEC.md
sed -n '1,220p' docs/E2W_PROJECT_LEDGER.md
sed -n '1,220p' STATUS.md
```

Use them as follows:

- `docs/E2W_SPEC.md`: the single current runtime/spec source.
- `docs/E2W_PROJECT_LEDGER.md`: durable project continuity, boundary decisions, and claim state.
- `STATUS.md`: current operational status and blockers.
- `docs/archive/`: historical references only. Archived docs are not current constraints.

If an archived doc conflicts with `docs/E2W_SPEC.md`, the current spec wins.

## 1. Project Identity

E2W studies counterfactual video editing with an explicit intervention contract:

```text
semantic intervention
+ first-frame-edited VACE conditioning video
+ explicit counterfactual description
+ grounded causal regions
+ operation-conditioned renderer
= counterfactual edited video
```

Do not recast E2W as:

- generic video editing;
- pure text-to-video generation;
- proof that the model has learned physics;
- a project where interface success is enough.

Reviewer boundary:

> VOID's unit of supervision is a deleted video; E2W's unit of supervision is an intervention-conditioned counterfactual contract linking planner state, causal regions, operation control, and rendered video.

## 2. Current VACE Runtime Contract

The current VACE runtime inputs are exactly:

```text
vace_conditioning_video
quadmask_npy
generation_mask
operation
vace_prompt
frame_num
```

Hard rules:

- Do not use `src_video`, `source_video`, `original_video`, or `factual_source_video` as VACE runtime inputs.
- `vace_conditioning_video` is the only visual condition passed to VACE.
- `vace_conditioning_video` must be first-frame-edited conditioning video.
- `generation_mask` is a unified full-domain all-generate mask, normally all `255` over `[T,H,W]`.
- `generation_mask` carries no E2W semantic edit meaning.
- Region semantics live only in `quadmask_npy`.
- Text input is named only `vace_prompt` in current docs/reports.
- Training manifests, target videos, side packets, and cycle roles are not defined by the current runtime spec.

Do not introduce parallel runtime contracts in new docs, tests, or reports.

## 3. Quadmask Semantics

`quadmask_npy` is the authoritative semantic control mask:

```text
dtype: uint8
shape: [T,H,W]
allowed values: {0,63,127,255}
```

Values:

- `0`: Q0 primary target or insertion region
- `63`: Q1 primary and affected overlap
- `127`: Q2 affected non-target region
- `255`: Q3 keep region

Rules:

- MP4/image previews are not authoritative masks.
- Resizing must preserve exact values, normally with nearest-neighbor semantics.
- Any frame/shape alignment must be explicit and recorded.
- Never collapse Q0/Q1/Q2/Q3 into a binary semantic mask.

## 4. Evidence Ladder

Classify every result by evidence level:

```text
INTERFACE  = files/commands/metadata exist and run
STRUCTURAL = shapes/values/paths prove signals enter the intended path
TRAINING   = loss/gate/gradient evidence from a real training run
CONTROL    = perturbation or swap shows response to operation/quadmask
VISUAL     = human/model review confirms edited video satisfies target criteria
RESEARCH   = ablation-backed evidence supports a paper-level claim
```

Do not upgrade evidence level without matching proof.

Examples:

- `edited_video` exists: INTERFACE.
- `quadmask_npy` has values `{0,63,127,255}` and reaches renderer metadata: STRUCTURAL.
- add/remove swap changes output consistently: CONTROL.
- human review confirms correct object addition/removal, Q2 response, and Q3 preservation: VISUAL.

## 5. Environment and Roots

Work from:

```text
/home/cwx/E2W
```

Use:

```text
/data/cwx/conda/envs/edit2world-phase1-real/bin/python
```

Large files and durable artifacts belong under:

```text
/data/cwx/E2W
```

Common roots:

```text
/data/cwx/E2W/runs
/data/cwx/E2W/data
/data/cwx/E2W/checkpoints
```

Do not place large artifacts in the git workspace unless explicitly requested.

Before real CUDA runs, check GPU contention:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Never kill other users' GPU jobs.

## 6. Validation Commands

For doc/spec-only changes, verify references and git diff; tests are usually not required unless you changed executable contracts.

For code or pipeline-contract changes, run relevant static tests:

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v03_quad_vace_contracts.py
```

If a test fails, fix the data, prompt, parser, planner, runner, or test expectation to match the accepted spec. Do not weaken tests merely to make a run pass.

## 7. Research Hygiene

Do not:

- claim visual/control/research success from file existence, non-black checks, or interface metadata;
- treat historical runs as current evidence without saying they are historical;
- treat teacher/manual artifacts as learned planner evidence;
- treat toy/debug runs as research evidence;
- use archived docs as current constraints;
- silently rewrite target-contaminated prompts;
- weaken schema, validators, prompt contracts, or evidence standards to pass a run;
- make source/original video a VACE runtime input under another name.

Do:

- ground current-state claims in `docs/E2W_SPEC.md`, ledger, `STATUS.md`, git, artifacts, tests, or run metadata;
- show evidence: commands run, paths inspected, test output, metadata values;
- keep current vs historical and interface vs learned behavior separate;
- update `docs/E2W_PROJECT_LEDGER.md` only for durable project facts, boundary decisions, canonical artifacts, or stable next-step changes.

## 8. Artifact and Git Policy

- Do not commit generated media, `.npy`, checkpoints, model weights, caches, or run directories.
- Commit code/docs/tests only.
- Prefer small commits with a clear Chinese message when working for Simon.
- Show `git status --short --branch` before and after meaningful changes.
- If the user asks only for investigation, do not edit files unless explicitly authorized.

## 9. Current Spec Invariants to Protect

These invariants should not be changed accidentally:

1. `docs/E2W_SPEC.md` is the single current runtime/spec source.
2. VACE runtime receives only `vace_conditioning_video`, `quadmask_npy`, `generation_mask`, `operation`, `vace_prompt`, and `frame_num`.
3. VACE runtime does not receive original/source/factual video.
4. `generation_mask` is full-domain and non-semantic.
5. `quadmask_npy` is the only semantic region contract.
6. `vace_prompt` is the only current prompt field name.
7. Training-data contract is intentionally unspecified until separately designed.
8. Interface success is never enough for visual/control/research claims.
