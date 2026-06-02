# Agent Instructions

This repo is the canonical code/docs/tests root for the artifact-first E2W smoke pipeline and v0.3 quadmask VACE contract. Prefer exact local commands, logs, manifests, and reports over generic explanations.

## Project Status

Last updated: 2026-06-02 UTC.

State:

- `/home/cwx/E2W` is the canonical code, Git, docs, tools, and tests repo.
- `/data/cwx/E2W` is the canonical durable data, checkpoints, runs, outputs,
  external assets, HF cache links, and tmp root.
- `/data/cwx/E2W/tools`, `/data/cwx/E2W/tests`, `/data/cwx/E2W/README.md`,
  and `/data/cwx/E2W/AGENTS.md` are compatibility symlinks back to the
  canonical home repo after migration.
- v0.2 pipeline correctness hardening is frozen as the current downstream smoke baseline, but the canonical forward pass now requires SFT VLM planner inference.
- v0.3 is the current VACE contract update: VACE can consume `quadmask.npy`
  through the E2W quad runner, with `--quadmask_npy`, `--operation`, and a
  binary generation mask.
- The canonical planner I/O schema is `e2w.planner_io.v6_executable.v1`.
  It supports both add and remove, and it requires executable grounding
  fields, not the old empty `quadmask_spec.primary/affected/keep` prompt.
- A previous full 8-sample run completed before the strict VACE prompt
  hard-fail contract was added. Treat it as a reference artifact, not as proof
  that the current stricter pipeline passes.

Goal:

- One fresh, artifact-first forward pass over the 8 smoke samples using the SFT VLM planner.
- Every run regenerates planner, mask, Qwen first-frame, VACE, package, and
  freshness artifacts.
- Planner regeneration means `eval_vlm_planner.py` writes `raw_output.txt` and
  `raw.pred.json` for every selected sample. Missing raw planner prediction
  artifacts means the run is invalid.
- v0.3 VACE reports must distinguish saved quadmask artifacts from true
  backend consumption of quadmask.
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
- VACE v0.3 uses the E2W quad runner. It consumes quadmask through an expanded
  VACE control/context path. The added channels are zero-initialized unless a
  finetuned adapter/checkpoint is explicitly provided, so interface success is
  not learned visual control.
- No VLM judge or automatic visual scoring is in scope yet.
- CUDA must be visible for real SAM2/Qwen/VACE runs; choose a free GPU and do
  not kill other users' jobs.

Current Results:

- Fresh SFT VLM planner run:
  `/data/cwx/E2W/runs/e2w_vlm_remove8_full_20260602T134159Z`.
- That run did use the SFT VLM planner checkpoint and wrote `raw_output.txt`
  plus `raw.pred.json` for all 8 smoke samples.
- It failed at the planner stage by contract before mask/Qwen/VACE:
  `quadmask_spec_executable: 0/8`, `primary_bbox_valid: 0/8`,
  `primary_point_valid: 0/8`, and `affected_grid_valid: 0/8`.
- The failure is expected strict behavior. Do not continue to mask/Qwen/VACE
  from this run unless planner output normalization/schema support is fixed.
- Reference run:
  `/data/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z`.
- That reference run reported 8/8 system interface ok, 8/8 Qwen interface ok,
  8/8 VACE completed, 0 black-frame failures, 0 resize mismatches, and Qwen
  visual review still `unreviewed`.
- That reference run is not proof of SFT VLM planner quality because its planner
  stage did not write `raw_output.txt` or `raw.pred.json`.
- After the strict VACE prompt hard-fail change, static tests pass:
  `/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py`
  reports 8 tests OK.
- Strict prompt status on current old labels:
  `0341` exports successfully and preserves `one potato`; `0076`, `0077`, and
  `0128` fail planner/export with `VacePromptContractError` because their
  candidate VACE text mentions the removed target or target subparts/materials.
- v0.3 contract docs and direct runner define quadmask-consuming VACE:
  `docs/CONTRACT.md`, `tools/run_vace_v03_quad_experiment.py`,
  `tools/run_wan_vace_quad_i2v.py`, and `tools/e2w_vace_quad_i2v.py`.

Current Blocker:

- The SFT planner currently outputs old/non-executable grounding such as
  `bbox_2d` or rectangle coordinates because the old SFT train data and old
  eval prompts did not teach executable grounding. The strict quadmask builder
  expects normalized executable keyframes/points/grid fields. Convert the SFT
  train/eval data to `e2w.planner_io.v6_executable.v1` and retrain before the
  next trusted full forward pass.

Next Actions:

- Archive old SFT JSONL files, convert train/eval prompts and labels to
  `e2w.planner_io.v6_executable.v1`, and retrain the SFT LoRA on the converted
  train split. Do not mix eval/smoke rows into train.
- Re-evaluate the retrained planner on 30 eval and 8 smoke samples. The planner
  stage must produce complete top-level JSON and executable quadmask specs.
- Re-run the 8 remove smoke samples. Only after planner passes should mask,
  Qwen first-frame, VACE, package/report, and any add-on experiment run.

## Environment

- Work from `/home/cwx/E2W` for code, tests, docs, and Git.
- After completing any feature edit or meaningful step of changes, create a focused git commit to mark that change. Do not bundle unrelated or user-owned work into the commit.
- Do not weaken, delete, or bypass schema/prompt/parser tests to make a run
  pass. If `tests/test_v02_contracts.py` fails, fix the data, prompt, parser,
  planner, or runner contract instead.
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

The canonical v0.2 full-smoke entrypoint is:

```bash
cd /home/cwx/E2W
RUN_NAME=e2w_v0_2_forward_$(date -u +%Y%m%dT%H%M%SZ)
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/run_v02_qwen_vace_smoke.py \
  --run-dir "/data/cwx/E2W/runs/$RUN_NAME" \
  --input-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl \
  --planner-adapter /data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v5_split_eval \
  --base-model /data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct \
  --operation remove \
  --cuda-visible-devices 4 \
  --run-vace
```

Replace `4` with the selected free GPU.

The stage order is fixed:

1. SFT VLM planner inference
2. quadmask build
3. Qwen first-frame edit
4. VACE v0 generation
5. package/report
6. artifact freshness audit

The planner stage must create `planner_pred/<sample-id>/raw_output.txt` and
`planner_pred/<sample-id>/raw.pred.json`. If either is missing for any selected
sample, the run did not use the SFT VLM planner and must not be treated as a
valid forward pass.

The input JSONL user prompts must use `e2w.planner_io.v6_executable.v1` and
must ask for:

- `quadmask_spec.operation`
- `primary.keyframes[].bbox_xyxy_norm1000`
- `primary.keyframes[].positive_points_norm1000`
- `affected.grid_shape`
- `affected.frame_ranges[].cells`

The old empty `{"quadmask_spec": {"primary": {}, "affected": {}, "keep": {}}}`
schema is archive-only.

## Planner Data Conversion And Retrain

Existing v6 eval/smoke files can be prompt-rewritten and validated with:

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/rewrite_planner_user_prompt_schema.py \
  --input-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl \
  --output-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl \
  --archive-dir /data/cwx/E2W/data/physics_iq_vlm_sft/archive/v6_prompt_rewrite_20260602 \
  --validate-assistant-executable
```

Generate train_v6 with visual grounding before retraining:

```bash
cd /home/cwx/E2W
OPENROUTER_API_KEY=<key> \
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/relabel_quadmask_specs_with_grounding.py \
  --input-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_train.jsonl \
  --output-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_train_v6_teacher_grounded.jsonl \
  --debug-dir /data/cwx/E2W/data/physics_iq_vlm_sft/grounding_debug/train_v6_teacher_grounded \
  --model qwen/qwen3.5-plus-20260420 \
  --grid-size 8 \
  --keep-going \
  --skip-existing
```

Only rows accepted by executable validation enter the train output. Review
queue rows are not training data.

Retrain:

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/train_qwen25vl_lora_sft.py \
  --train-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_train_v6_teacher_grounded.jsonl \
  --eval-jsonl /data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v6_teacher_grounded.jsonl \
  --base-model /data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct \
  --output-dir /data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v6_executable \
  --max-steps 68 \
  --save-steps 68 \
  --eval-steps 68
```

## v0.3 Quadmask VACE

The canonical direct v0.3 quadmask VACE wrapper is:

```bash
cd /home/cwx/E2W
RUN_DIR=/data/cwx/E2W/runs/e2w_v0_3_quad_vace_add_0076_$(date -u +%Y%m%dT%H%M%SZ)
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/run_vace_v03_quad_experiment.py \
  --src-video /home/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z/0076/edited_video.mp4 \
  --prompt "Add a yellow mug on the rotating turntable in front of the spotlight" \
  --quadmask-npy /home/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z/0076/quadmask.npy \
  --operation add \
  --run-dir "$RUN_DIR" \
  --generation-mask-mode quadmask-editable \
  --align-quadmask nearest \
  --cuda-visible-devices 4 \
  --sample-steps 8 \
  --run-vace
```

Use `--align-quadmask nearest` only when the mismatch is intentional and the
metadata records the deterministic conversion. Silent frame-count mismatch is
not allowed in v0.3.

## Reporting Rules

- Interface success is not visual success.
- Keep `qwen_visual_review_status` as `unreviewed` unless a human has inspected the artifact.
- Do not report size checks, black-frame checks, or Qwen interface checks as Qwen visual correctness.
- If VACE is requested with `--run-vace` and the subprocess returns nonzero, status must be `failed`, not `prepared`.
- Do not report `quadmask_consumed_by_backend: true` for legacy `run_vace_v0.py`.
- For v0.3, `quadmask_passed_to_backend_command: true` requires a command
  containing `--quadmask_npy`, `--operation`, and `--generation_mask`.
- For v0.3, `quadmask_consumed_by_backend: true` additionally requires the
  subprocess to return `0` and produce `edited_video.mp4`.
- Attribute failures from the logs and manifest. Start with `/data/cwx/E2W/runs/<run-name>/stage_logs/*.stderr.txt`, `manifest.jsonl`, `report.md`, and `artifact_freshness.json`.

Required machine gates for a clean forward pass:

- planner/mask/first_frame/vace entries are all present for all 8 samples
- planner entries include `raw_output.txt` and `raw.pred.json` for all 8 samples
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
- Protected objects come from the raw planner prediction and stay in keep constraints, not in the remove target line.
- `expected_background` is compatibility metadata. Use local-fill wording for Qwen: fill target pixels with plausible local background.
- VACE prompts must stay neutral and target-free; no `remove/delete/erase/removed` wording.
- VACE prompts must also avoid visible target subparts/materials, including negative wording like `no <target>`.
- If candidate planner text for a VACE prompt mentions the removed target or target subparts, fail the planner/export stage with a contract error. Do not silently drop the sentence or use a neutral fallback.
- If no target-free semantic line remains for the VACE prompt, fail instead of using generic neutral text.
- Preserve clear counts for visible non-target objects when they matter, such as `one potato`.
- Qwen-Image-Edit consumes image + text prompt only. The target mask is for QC/debug and metadata records `target_mask_consumed_by_backend: false`.
- v0.3 VACE consumes `quadmask.npy` with values `0/63/127/255`:
  primary, primary+affected overlap, affected, keep.
- v0.3 binary generation mask remains only a known/generate gate:
  `0` keep/known, `255` generate.
- v0.3 does not imply trained quadmask semantics unless a trained adapter or
  checkpoint is explicitly recorded.

## Validation

Run static checks after pipeline changes:

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py
```

A known-good full run is:

`/data/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z`
