# Agent Instructions

This repo is the canonical code/docs/tests root for the artifact-first E2W v0.2 smoke pipeline. Prefer exact local commands, logs, manifests, and reports over generic explanations.

## Project Status

Last updated: 2026-06-02 UTC.

State:

- `/home/cwx/E2W` is the canonical code, Git, docs, tools, and tests repo.
- `/data/cwx/E2W` is the canonical durable data, checkpoints, runs, outputs,
  external assets, HF cache links, and tmp root.
- `/data/cwx/E2W/tools`, `/data/cwx/E2W/tests`, `/data/cwx/E2W/README.md`,
  and `/data/cwx/E2W/AGENTS.md` are compatibility symlinks back to the
  canonical home repo after migration.
- v0.2 pipeline correctness hardening is in progress.
- A previous full 8-sample run completed before the strict VACE prompt
  hard-fail contract was added. Treat it as a reference artifact, not as proof
  that the current stricter pipeline passes.

Goal:

- One fresh, artifact-first v0.2 forward pass over the 8 smoke samples.
- Every run regenerates planner, mask, Qwen first-frame, VACE, package, and
  freshness artifacts.
- VACE prompt text is target-free, count-aware for visible non-target objects,
  and fails hard instead of falling back when planner text violates the
  contract.
- Reports separate interface success from visual success. Human visual review
  remains explicit and is not inferred from size checks.

Assumptions:

- Qwen-Image-Edit consumes image + text prompt only; target masks are QC/debug
  artifacts and metadata records `target_mask_consumed_by_backend: false`.
- VACE v0 uses the prompt plus conditioning/generation-mask inputs; quadmask
  frame-count mismatch is a warning in v0, not a hard failure.
- No VLM judge or automatic visual scoring is in scope yet.
- CUDA must be visible for real SAM2/Qwen/VACE runs; choose a free GPU and do
  not kill other users' jobs.

Current Results:

- Reference run:
  `/data/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z`.
- That reference run reported 8/8 system interface ok, 8/8 Qwen interface ok,
  8/8 VACE completed, 0 black-frame failures, 0 resize mismatches, and Qwen
  visual review still `unreviewed`.
- After the strict VACE prompt hard-fail change, static tests pass:
  `/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py`
  reports 8 tests OK.
- Strict prompt status on current old labels:
  `0341` exports successfully and preserves `one potato`; `0076`, `0077`, and
  `0128` fail planner/export with `VacePromptContractError` because their
  candidate VACE text mentions the removed target or target subparts/materials.

Current Blocker:

- Regenerate or manually fix planner labels for `0076`, `0077`, and `0128`
  before the next full v0.2 forward pass. Do not bypass this by sentence
  dropping or neutral fallback.

Next Actions:

- Fix the hard-fail planner labels or planner prompt output for the blocked
  samples.
- Re-run planner export for the 8 smoke samples and confirm VACE prompt
  contract passes before GPU stages.
- Then run the full forward pass and update this Project Status section with
  the new run directory, gates, and remaining review state.

## Environment

- Work from `/home/cwx/E2W` for code, tests, docs, and Git.
- Write full-run artifacts under `/data/cwx/E2W/runs`.
- Use `/data/cwx/conda/envs/edit2world-phase1-real/bin/python`.
- Do not assume bare `python` has the right dependencies.
- For real SAM2/Qwen/VACE execution, CUDA must be visible. In the default Codex sandbox, CUDA may be hidden; request outside-sandbox execution when needed.
- Before running a full forward pass, check GPU contention with:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Do not kill other users' GPU jobs. Pick an emptier GPU and pass it explicitly through `--cuda-visible-devices`.

## Canonical Forward Pass

The canonical v0.2 entrypoint is:

```bash
cd /home/cwx/E2W
RUN_NAME=e2w_v0_2_forward_$(date -u +%Y%m%dT%H%M%SZ)
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/run_v02_qwen_vace_smoke.py \
  --run-dir "/data/cwx/E2W/runs/$RUN_NAME" \
  --cuda-visible-devices 4 \
  --run-vace
```

Replace `4` with the selected free GPU.

The stage order is fixed:

1. planner export
2. quadmask build
3. Qwen first-frame edit
4. VACE v0 generation
5. package/report
6. artifact freshness audit

Do not use `tools/create_v02_qwen_vace_smoke_bundle.py` as the v0.2 validation path. It is legacy/deprecated. v0.2 validation requires freshly regenerated planner, mask, Qwen, and VACE artifacts.

## Reporting Rules

- Interface success is not visual success.
- Keep `qwen_visual_review_status` as `unreviewed` unless a human has inspected the artifact.
- Do not report size checks, black-frame checks, or Qwen interface checks as Qwen visual correctness.
- If VACE is requested with `--run-vace` and the subprocess returns nonzero, status must be `failed`, not `prepared`.
- Attribute failures from the logs and manifest. Start with `/data/cwx/E2W/runs/<run-name>/stage_logs/*.stderr.txt`, `manifest.jsonl`, `report.md`, and `artifact_freshness.json`.

Required machine gates for a clean forward pass:

- planner/mask/first_frame/vace entries are all present for all 8 samples
- `system interface ok: 8/8`
- `qwen_image_edit interface ok: 8/8`
- `VACE completed: 8/8`
- `black-frame failures: 0`
- `resize mismatches: 0`
- `artifact_freshness.json` has `all_critical_stages_regenerated: true`
- `artifact_freshness.json` has `no_other_run_symlinks: true`

## Contract Boundaries

- Qwen prompts must explicitly name the target label, aliases, and visual descriptor.
- Qwen prompts must not use `primary subject`.
- Protected objects come from the raw teacher/planner labels and stay in keep constraints, not in the remove target line.
- `expected_background` is compatibility metadata. Use local-fill wording for Qwen: fill target pixels with plausible local background.
- VACE prompts must stay neutral and target-free; no `remove/delete/erase/removed` wording.
- VACE prompts must also avoid visible target subparts/materials, including negative wording like `no <target>`.
- If candidate planner text for a VACE prompt mentions the removed target or target subparts, fail the planner/export stage with a contract error. Do not silently drop the sentence or use a neutral fallback.
- If no target-free semantic line remains for the VACE prompt, fail instead of using generic neutral text.
- Preserve clear counts for visible non-target objects when they matter, such as `one potato`.
- Qwen-Image-Edit consumes image + text prompt only. The target mask is for QC/debug and metadata records `target_mask_consumed_by_backend: false`.

## Validation

Run static checks after pipeline changes:

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py
```

A known-good full run is:

`/data/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z`
