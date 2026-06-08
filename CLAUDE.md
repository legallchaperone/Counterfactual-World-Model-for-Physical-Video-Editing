# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is the canonical code/docs/tests root for the **E2W (Edit2World) smoke pipeline** and the **v0.3 quadmask VACE contract**. It is "artifact-first": correctness is judged from generated run artifacts (manifests, reports, freshness audits), not from prose claims.

Read `STATUS.md` for the current project state, goal, and blocker — **do not infer current status from any other file** (including this one). `docs/CONTRACT.md` is the authoritative runtime contract; `AGENTS.md` holds the same behavioral rules (largely in Chinese).

## Roots and environment

- **Code/tests/docs/git**: `/home/cwx/E2W` (work from here).
- **Data, checkpoints, runs, outputs, external assets, HF cache, tmp**: `/data/cwx/E2W` (reached via symlinks in the repo root: `data/`, `checkpoints/`, `runs/`, etc.). All of these are gitignored.
- **Always use this Python** — bare `python` will not have the right deps:
  ```
  /data/cwx/conda/envs/edit2world-phase1-real/bin/python
  ```
- Media/array artifacts (`*.mp4`, `*.png`, `*.npy`, `*.safetensors`, etc.) are gitignored; only code, docs, and `tests/fixtures/*.jsonl` are tracked.

## Commands

Run the static contract tests (do this for any pipeline change):
```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v03_quad_vace_contracts.py
```

Run a single test:
```bash
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest \
  tests.test_v02_contracts.V02PromptContractTests.test_vace_prompt_is_neutral_and_target_free
```

Before any real CUDA run, check GPU contention and pick a free index (**never kill other users' GPU jobs**):
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Full forward pass, single-test invocation patterns, and the planner data-conversion / retraining workflow are all spelled out with exact commands in `docs/CONTRACT.md` — prefer copying from there.

## Architecture

A planner produces structured JSON, which drives a chain of mask → first-frame edit → VACE generation → package/report stages. Each stage is its own `tools/*.py` script; an orchestrator runs them in order and writes a manifest + report + freshness audit.

- **`tools/e2w_v0_common.py`** — shared core (~1.3k lines). Owns the schema constants, the canonical planner prompt builder `build_planner_user_prompt`, JSON parsing (`parse_json_output`), contract normalization/validation, and `serialize_vace_prompt` (the VACE-prompt safety gate, raises `VacePromptContractError`). Most contract rules live here; the tests import these helpers directly.
- **`tools/run_v02_qwen_vace_smoke.py`** — the v0.2 orchestrator. Regenerates artifacts in a fixed order: `eval_vlm_planner` → `build_quadmask_from_spec` → `run_first_frame_edit` (backend `qwen_image_edit`) → `run_vace_v0` → `package_v02_qwen_vace_smoke` → `artifact_freshness.json`. Propagates `CUDA_VISIBLE_DEVICES` explicitly to each stage.
- **`tools/eval_vlm_planner.py`** — planner inference stage; loads Qwen2.5-VL-7B + LoRA adapter, writes `raw_output.txt` and `raw.pred.json` per sample. A run without those files for every sample is **not** a valid forward pass.
- **`tools/e2w_vace_quad_i2v.py`** + **`tools/run_wan_vace_quad_i2v.py`** — the v0.3 quad VACE backend. Expands the Wan-VACE control branch from legacy 96 channels to a 416-channel E2W context that consumes `quadmask.npy`. `tools/run_vace_v03_quad_experiment.py` is the direct experiment wrapper.
- **Data prep / training**: `relabel_quadmask_specs_with_grounding.py` (adds executable grounding via OpenRouter), `rewrite_planner_user_prompt_schema.py` (archives + rewrites JSONL prompts to v6), `train_qwen25vl_lora_sft.py` (LoRA SFT).

## Hard contract rules (do not weaken to make a run pass)

These are enforced by `tests/test_v02_contracts.py` / `test_v03_quad_vace_contracts.py`. If a test fails, fix the data, prompt, parser, planner, or runner — **never weaken the test**. Attribute planner failures from artifacts (`manifest.jsonl`, `planner_pred/*/planner_eval.json`, `raw_output.txt`, `raw.pred.json`, `report.md`, `artifact_freshness.json`).

- **Planner schema**: the only valid I/O schema for new train/eval/smoke/forward-pass data is `e2w.planner_io.v6_executable.v1`. The old empty `{"quadmask_spec": {"primary": {}, "affected": {}, "keep": {}}}` schema is archive-only. Planner output must be one complete top-level JSON object; nested target-object fragments are parse failures.
- **Planner prompt** is final-rule-only, built by `e2w_v0_common.build_planner_user_prompt`: role/schema → task operation → full schema JSON → current `video_id`/request → final rules. Do **not** embed one-shot wrapper examples (they made the LoRA copy wrapper keys).
- **Executable grounding** requires `quadmask_spec.primary.keyframes[].bbox_xyxy_norm1000`, `.positive_points_norm1000`, `affected.grid_shape`, and `affected.frame_ranges[].cells`.
- **VACE prompts must be target-free.** For remove tasks, `counterfactual_expectation.if_removed` is copied into the VACE prompt and must not name the target, aliases, visible subparts/materials, or negative forms (`no <target>`, `without <target>`, `<target> is removed/no longer present`, `where the X was`). If candidate text violates this, **hard-fail** (`VacePromptContractError`) — do not silently drop the sentence or fall back to neutral text. Preserve clear non-target counts explicitly (e.g. `one potato`).
- **Qwen first-frame prompts** name the real target label/aliases/visual descriptor (never `primary subject`) and only ask to fill target pixels with plausible local background, protecting non-target objects. Qwen-Image-Edit consumes image + text only; metadata records `target_mask_consumed_by_backend: false`.
- **Interface success ≠ visual success.** Do not infer visual quality from file size, black-frame checks, or interface checks. Keep `qwen_visual_review_status` = `unreviewed` until a human inspects artifacts.
- **v0.3 alignment**: `quadmask.npy` values are `0/63/127/255` = primary / primary+affected overlap / affected / keep. Frame/shape mismatches must be recorded explicitly (no silent mismatch). `quadmask_passed_to_backend_command: true` only when the command contains `--quadmask_npy`, `--operation`, and `--generation_mask`; `quadmask_consumed_by_backend: true` only when that command ran successfully and produced `edited_video.mp4`.
