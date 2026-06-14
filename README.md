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

- Counterfactual Planner is the current correct planner design.
- Archived executable-planner artifacts are archived references only, not current baselines.
- The main blocker is making the Counterfactual Planner grounding bridge and runtime adapter conform to the current spec.
- The VACE control-branch training path is separate and still needs corrected real training plus control/visual validation.
- The add pipeline now has an **INTERFACE-level** smoke success.

Verified add INTERFACE run:

```text
/data/cwx/E2W/runs/add_pipeline_interface_add_bg_000001_20260609T024340Z
```

What that run proves:

- original video + user prompt reached actual planner/model inference;
- `vace_prompt` came from the planner/model and was passed through unchanged;
- Qwen Edit produced an edited first frame;
- SAM2 produced an add quadmask from the edited first frame;
- `quadmask.npy` and full-domain `generation_mask` had correct shape/value contracts;
- VACE produced a non-empty `edited_video.mp4`.

What it does **not** prove:

- visual quality;
- learned planner add quality;
- learned VACE add semantics;
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

## Running the Current Add Interface Smoke

Use this only as an interface check, not as a visual-quality benchmark.

```bash
cd /home/cwx/E2W
CUDA_VISIBLE_DEVICES=<gpu> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/e2w_add.py \
  --planner-adapter <add-planner-lora> \
  --cuda-visible-devices <gpu> \
  --frame-num 21 \
  --planner-attempts 3 \
  --qwen-steps 5 \
  --vace-sample-steps 2
```

The three unified interfaces are `tools/e2w_remove.py`, `tools/e2w_add.py`, and
`tools/e2w_add_then_remove.py`, all built on `tools/e2w_pipeline_core.py`. The add
interface requires an add-trained planner adapter (`--planner-adapter`); there is no
archived default.

Unit test:

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests.test_add_pipeline_interface
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
