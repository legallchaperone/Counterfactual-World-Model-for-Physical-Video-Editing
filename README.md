# E2W: Edit2World Counterfactual Video Editing

E2W studies **counterfactual video editing with an explicit intervention contract**:

```text
semantic intervention
+ first-frame-edited VACE conditioning video
+ explicit counterfactual description
+ grounded causal regions
+ operation-conditioned renderer
= counterfactual edited video
```

E2W is **not** a generic video editing benchmark, not pure text-to-video generation, and not a claim that the current system already learned a physical world model.

## Read First

These files are the current source of truth:

```text
docs/E2W_SPEC.md              # single current runtime contract
STATUS.md                     # current operational status
docs/E2W_PROJECT_LEDGER.md    # durable project ledger / continuity notes
AGENTS.md                     # agent operating manual
```

Historical specs, handoffs, and old runbooks live under `docs/archive/` and are reference-only. If an archived file or an old README-era command disagrees with `docs/E2W_SPEC.md`, the current spec wins.

Canonical locations on `cwx`:

```text
repo:      /home/cwx/E2W
artifacts: /data/cwx/E2W
python:    /data/cwx/conda/envs/edit2world-phase1-real/bin/python
```

## Current Runtime Contract

The VACE runtime receives exactly these E2W-controlled inputs:

```text
vace_conditioning_video
quadmask_npy
generation_mask
operation
vace_prompt
frame_num
```

Do **not** introduce parallel runtime contracts in new code, docs, or reports.

Important rules:

- `vace_conditioning_video` is the only visual condition passed to VACE.
- `vace_conditioning_video` is a first-frame-edited conditioning video, not the original observed video.
- No `src_video`, `source_video`, `original_video`, or `factual_source_video` field is part of the E2W VACE runtime contract.
- `quadmask_npy` is the semantic E2W control mask.
- `generation_mask` is full-domain all-generate for normal E2W runs and carries no E2W semantic meaning.
- `operation` is explicit: `remove` or `add`.
- `vace_prompt` must come from the actual upstream planner/model inference path for the run; runners must not hard-code, rewrite, or teacher/manual-substitute it.

See `docs/E2W_SPEC.md` for the full contract.

## Quadmask Semantics

`quadmask_npy` is authoritative:

```text
dtype: uint8
shape: [T, H, W]
allowed values: {0, 63, 127, 255}
```

| value | region | meaning |
|---:|---|---|
| `0` | Q0 | primary target or insertion region |
| `63` | Q1 | primary and affected overlap |
| `127` | Q2 | affected non-target region |
| `255` | Q3 | keep region |

Preview images/videos are not authoritative because compression can perturb exact values.

## Evidence Ladder

Every result must be described with an evidence level:

```text
INTERFACE  = files/commands/metadata exist and run
STRUCTURAL = shapes/values/paths prove signals enter the intended model path
TRAINING   = loss/gate/gradient evidence from a real training run
CONTROL    = perturbation or swap proves response to operation/quadmask
VISUAL     = review confirms the edited video satisfies target criteria
RESEARCH   = ablation-backed evidence supports a paper-level claim
```

Do not upgrade claims without matching evidence. In particular, `edited_video.mp4` existence is INTERFACE, not visual success.

## Current Status Snapshot

For current details, read `STATUS.md`. At a high level:

- v0.2/v7 executable planner route: remove8 gate still fails strict downstream requirements.
- v8 planner-text route: strong target-free text/schema results, but it does not output executable quadmask grounding.
- VACE Phase 1A control-branch work is separate from planner training and still needs corrected real 14B reruns plus control/visual validation.
- Add pipeline has a current-spec INTERFACE smoke success, not visual/control/research success.

Verified add INTERFACE run:

```text
/data/cwx/E2W/runs/add_pipeline_interface_add_bg_000001_20260609T024340Z
```

That run used real planner/model inference for `vace_prompt`, Qwen Edit for `edited_first_frame`, SAM2 on `edited_first_frame` for add quadmask, full-domain all-255 `generation_mask`, and VACE to produce `edited_video.mp4`. It does **not** prove visual quality, learned planner add quality, or learned VACE add semantics.

## Useful Commands

Check repository state:

```bash
cd /home/cwx/E2W
git status --short --branch
git log -5 --oneline
```

Run the add INTERFACE unit tests:

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests.test_add_pipeline_interface
```

Run the add INTERFACE smoke when a suitable GPU is free:

```bash
cd /home/cwx/E2W
CUDA_VISIBLE_DEVICES=<gpu> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/run_add_pipeline_interface.py \
  --cuda-visible-devices <gpu> \
  --frame-num 21 \
  --planner-attempts 3 \
  --qwen-steps 5 \
  --vace-sample-steps 2
```

The add runner is an INTERFACE smoke runner. It must not be used to claim visual success without separate review.

## Current Do-Not-Do List

- Do not pass original/source/factual video to VACE as a runtime input.
- Do not use `src_video` as a current E2W runtime field name; it may appear only as a legacy backend adapter argument when explicitly documented.
- Do not use `generation_mask` to encode Q0/Q1/Q2/Q3 semantics.
- Do not silently rewrite invalid planner outputs.
- Do not use teacher/manual `vace_prompt` for current pipeline acceptance runs.
- Do not report interface success as visual/control/research success.
- Do not revive archived docs as current constraints.

## Worktree / Branch Hygiene

Use feature worktrees for isolated development, then merge via PR into `main`. After merge, remove the feature worktree/branch unless it remains active.

Current local-only worktrees may exist for unfinished research branches; check them before assuming all work is on GitHub:

```bash
git worktree list --porcelain
git branch -vv --all
```
