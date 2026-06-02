# E2W v0.2 Forward Pass

This repo currently standardizes the v0.2 Qwen + VACE smoke pipeline for one full forward pass over the 8 primary smoke samples.

Canonical code root: `/home/cwx/E2W`.

Canonical data/artifact root: `/data/cwx/E2W`.

Canonical successful run for comparison:

`/data/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z`

That run completed planner, mask, Qwen first-frame edit, VACE generation, package, and artifact freshness checks. Its report shows:

- samples: 8
- system interface ok: 8/8
- qwen_image_edit interface ok: 8/8
- VACE completed: 8/8
- black-frame failures: 0
- resize mismatches: 0
- Qwen visual review status: `unreviewed` for all samples

Interface checks are not visual success checks. Human review must inspect the generated artifacts before marking visual success.

## Full Forward Pass

Run from `/home/cwx/E2W` with the project env Python. Pick a free GPU first:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Then run the whole v0.2 chain:

```bash
cd /home/cwx/E2W
RUN_NAME=e2w_v0_2_forward_$(date -u +%Y%m%dT%H%M%SZ)
/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/run_v02_qwen_vace_smoke.py \
  --run-dir "/data/cwx/E2W/runs/$RUN_NAME" \
  --cuda-visible-devices 4 \
  --run-vace
```

Replace `4` with the free physical GPU index. In Codex, the default sandbox may hide CUDA devices; request outside-sandbox execution for the real forward pass if `torch.cuda.is_available()` is false or `/dev/nvidia*` is missing.

The orchestrator regenerates artifacts in this order:

1. `export_teacher_grounded_bundle.py`
2. `build_quadmask_from_spec.py`
3. `run_first_frame_edit.py --backend qwen_image_edit`
4. `run_vace_v0.py --run-vace`
5. `package_v02_qwen_vace_smoke.py`
6. `artifact_freshness.json`

Do not use `tools/create_v02_qwen_vace_smoke_bundle.py` for v0.2 validation. It is retained only as a legacy/deprecated helper and must not be used as the canonical entrypoint because v0.2 requires freshly regenerated planner, mask, Qwen, and VACE artifacts.

## Required Gates

After the run finishes, inspect:

```bash
RUN_DIR=/data/cwx/E2W/runs/$RUN_NAME
cat "$RUN_DIR/report.md"
cat "$RUN_DIR/artifact_freshness.json"
```

Required machine gates:

- `artifact_freshness.json`: `all_critical_stages_regenerated: true`
- `artifact_freshness.json`: `no_other_run_symlinks: true`
- manifest contains 8 `planner_eval`, 8 `mask_builder`, 8 `first_frame`, and 8 `vace_v0` entries
- report shows `system interface ok: 8/8`
- report shows `qwen_image_edit interface ok: 8/8`
- report shows `VACE completed: 8/8`
- report shows `black-frame failures: 0`
- report shows `resize mismatches: 0`
- report keeps visual review as `unreviewed` until a human checks the images/videos

Useful per-sample artifacts live under:

`/data/cwx/E2W/runs/<run-name>/<sample-id>/`

Expected files include Qwen prompt, edited first frame, VACE prompt, VACE conditioning video, VACE generation mask, edited video, metadata, and contact sheets.

## Contract Notes

- Qwen first-frame prompts name the actual target label, aliases, and visual descriptor. They must not use `primary subject`.
- Qwen prompts protect non-target objects and only ask to fill target pixels with plausible local background.
- Qwen-Image-Edit currently consumes image + text prompt only. `first_frame_edit_metadata.json` records `target_mask_consumed_by_backend: false`.
- VACE prompts are neutral continuity prompts. They must not contain target aliases or removal wording such as `remove`, `delete`, `erase`, or `removed subject`.
- VACE prompts must not mention the removed target or visible target subparts/materials, even as negative wording such as `no <target>`.
- If candidate planner text for a VACE prompt mentions the removed target or target subparts, the planner/export stage must fail with a contract error. Do not silently drop the sentence or use a neutral fallback.
- If no target-free semantic line remains for the VACE prompt, fail instead of using generic neutral text.
- When a visible non-target object count is clear and relevant, preserve it explicitly in planner/VACE text, for example `one potato`.
- v0 records quadmask/frame-count mismatches as warnings. Do not treat those warnings as visual success or failure unless the stage contract changes.

## Static Validation

Run:

```bash
cd /home/cwx/E2W
/data/cwx/conda/envs/edit2world-phase1-real/bin/python -m unittest tests/test_v02_contracts.py
```

This checks prompt contracts, report semantics, VACE failure attribution, and explicit `CUDA_VISIBLE_DEVICES` propagation in the canonical runner.
