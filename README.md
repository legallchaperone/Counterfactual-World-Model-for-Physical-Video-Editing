# Edit2World (E2W)

E2W is a research prototype for **counterfactual video editing under an explicit intervention contract**.

Given an observed video and a user intervention such as "remove the object" or "add a mug", E2W separates the problem into explicit pieces:

```text
user intervention
→ Counterfactual Planner output
→ grounding bridge
→ counterfactual first frame
→ causal-region quadmask
→ operation-conditioned video renderer
→ edited video
```

The project goal is not just to make an edited clip. The research question is whether making the intervention contract explicit — semantic state, causal regions, operation, and renderer conditioning — can support more testable counterfactual video editing than treating the task as ordinary video editing or pure text-to-video generation.

E2W does **not** currently claim that the system has learned a physical world model. Every result should be reported with the evidence level it actually supports.

## What to Read

For a quick human orientation, start here. For operational details, use the canonical docs:

| File | Audience | Purpose |
|---|---|---|
| `README.md` | humans | project overview, mental model, current state |
| `docs/E2W_SPEC.md` | humans + agents | single current runtime contract |
| `STATUS.md` | humans + agents | current operational status and blockers |
| `docs/E2W_PROJECT_LEDGER.md` | humans + agents | durable project decisions and evidence ledger |
| `docs/UNIFIED_PIPELINE_RUNBOOK.md` | humans + agents | fixed current commands for remove/add/add-then-remove |
| `AGENTS.md` | agents | rules for coding/research agents working in this repo |

Historical specs, handoffs, and old runbooks live under `docs/archive/`. They are useful for archaeology, but not current constraints.

## Core Idea

E2W represents an edit as an intervention contract:

```text
semantic intervention
+ first-frame-edited VACE conditioning video
+ explicit counterfactual description
+ grounded causal regions
+ operation-conditioned renderer
= counterfactual edited video
```

This means the video renderer should not receive the original video as a hidden source condition. The renderer gets a counterfactual conditioning video plus structured controls. The original video can still be used upstream for planner context, provenance, and audit.

## Runtime Contract in One Screen

The current planner contract is the Counterfactual Planner. Its compatible schema id remains `e2w.planner_output.v8_tool_augmented_grounding.v1` for existing artifacts.

```text
target_ref
edit_type
counterfactual_state
if_removed
```

The grounding bridge turns `target_ref` into `quadmask_npy`; `counterfactual_state` and `if_removed` become the basis for `vace_prompt`.

The current VACE runtime input contract is:

```text
vace_conditioning_video   # first-frame-edited conditioning video
quadmask_npy              # semantic region contract, uint8 [T,H,W]
generation_mask           # full-domain all-generate mask, non-semantic
operation                 # remove | add
vace_prompt               # produced by actual upstream planner/model inference
frame_num                 # VACE temporal length
```

The important boundary:

```text
original/source/factual video is not a VACE runtime input
```

See `docs/E2W_SPEC.md` for the precise rules.

## Quadmask

`quadmask_npy` is the semantic mask. It is authoritative; previews are not.

| value | region | meaning |
|---:|---|---|
| `0` | Q0 | primary target or insertion region |
| `63` | Q1 | primary and affected overlap |
| `127` | Q2 | affected non-target region |
| `255` | Q3 | keep region |

`generation_mask` is different: in current E2W runtime it is normally full-domain all `255` and carries no semantic edit meaning.

## Current State

See `STATUS.md` for the latest details. Short version:

- The current public interfaces are unified into three entry points: `tools/e2w_remove.py`, `tools/e2w_add.py`, and `tools/e2w_add_then_remove.py`.
- All three share `tools/e2w_pipeline_core.py` and emit the same six-input VACE runtime contract.
- Deprecated legacy names are shims only: `tools/run_counterfactual_planner_pipeline.py`, `tools/run_add_pipeline_interface.py`, and `tools/run_add_then_remove_pipeline.py`.
- Add is first-class: add planner output drives masked inpaint first-frame editing, then SAM2 grounds the inserted object on the edited first frame with a change-overlap consistency guard.
- Current add/add-then-remove runs are still **INTERFACE** evidence unless separately reviewed/tested. Control, visual, and research evidence are not established.

What an INTERFACE run proves:

- original/upstream context reached actual planner/model inference;
- `vace_prompt` came from the planner/model and was passed through unchanged;
- first-frame edit, `quadmask_npy`, full-domain `generation_mask`, and VACE invocation completed;
- metadata records the current six runtime inputs and alignment.

What it does **not** prove:

- visual quality;
- learned planner generalization;
- learned VACE add/remove semantics;
- physical correctness.

## Evidence Levels

Use these labels when discussing results:

```text
INTERFACE  = files/commands/metadata exist and run
STRUCTURAL = shapes/values/paths prove signals enter the intended path
TRAINING   = loss/gate/gradient evidence from a real training run
CONTROL    = perturbation or swap shows response to operation/quadmask
VISUAL     = review confirms the edited video satisfies target criteria
RESEARCH   = ablation-backed evidence supports a paper-level claim
```

A generated `edited_video.mp4` is INTERFACE evidence unless separately reviewed or tested.

## Fixed Runbook

The fixed current runbook is `docs/UNIFIED_PIPELINE_RUNBOOK.md`.

Minimal add smoke shape:

```bash
cd /home/cwx/E2W
export PY=/data/cwx/conda/envs/edit2world-phase1-real/bin/python
export ADD_PLANNER=/data/cwx/E2W/checkpoints/vlm_planner_lora_add_v1_20260615
export CONTROL_BRANCH=/data/cwx/E2W/checkpoints/v04_real_overfit_14b_specfix_selfinsert_20260612
CUDA_VISIBLE_DEVICES=<gpu> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY tools/e2w_add.py \
  --source-video <source_video.mp4> \
  --user-prompt "Add a red mug on the table near the center of the image." \
  --sample-id <sample_id> \
  --run-dir /data/cwx/E2W/runs/e2w_add_<sample_id> \
  --planner-adapter "$ADD_PLANNER" \
  --control-branch-checkpoint "$CONTROL_BRANCH" \
  --cuda-visible-devices <gpu>
```

Validation:

```bash
cd /home/cwx/E2W
$PY -m unittest discover -s tests -p 'test*.py'
```

Project locations on `cwx`:

```text
repo:      /home/cwx/E2W
artifacts: /data/cwx/E2W
python:    /data/cwx/conda/envs/edit2world-phase1-real/bin/python
```

## For Contributors

Before changing runtime contracts, read `docs/E2W_SPEC.md`. Before making claims about progress, read `STATUS.md` and `docs/E2W_PROJECT_LEDGER.md`.

For agents and automation, follow `AGENTS.md`. The README is intentionally a human-facing overview; detailed operating rules live in the spec, status, ledger, and agent manual.
