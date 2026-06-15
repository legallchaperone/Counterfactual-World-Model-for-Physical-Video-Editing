# E2W Artifact Index (runs / checkpoints)

> **Status snapshot: 2026-06-13.** This is a *readability index*, not a contract. It classifies the
> large artifacts under `/data/cwx/E2W/{runs,checkpoints}` so it's clear which are current evidence,
> which are kept for history, which are wired into current tools, and which are unreferenced cleanup
> candidates.
>
> Scope notes:
> - These directories are **gitignored** (`.gitignore` lists `/runs /checkpoints`) and are symlinks
>   into `/data`. Nothing here is in the git repo.
> - The authoritative current-evidence statements live in `STATUS.md` and `docs/E2W_PROJECT_LEDGER.md`.
>   If this index ever disagrees with them, they win.
> - "no tracked reference" = no hit in tracked `*.py`/`*.md` at the time of this snapshot. `/data` is
>   shared with git worktrees (`.worktree/`), which may reference paths independently — verify before
>   any deletion.
> - **Nothing has been moved or deleted.** Sizes are approximate (`du -sh`).

## Status legend

| Status | Meaning |
|--------|---------|
| **CURRENT** | Cited in STATUS/ledger as live evidence. Keep. |
| **CODE-DEFAULT** | Hardcoded as a default path in a *current* tool. Moving/deleting breaks the tool unless the code is edited first. |
| **ARCHIVED-REF** | Listed in STATUS/ledger/`docs/archive/` as a historical reference (kept intentionally for failure analysis). Superseded, not current. |
| **STALE-DOC** | Explicitly marked stale/superseded in STATUS/ledger but still referenced there. |
| **INVALIDATED** | Known-bad output that must not be used as evidence. |
| **CLEANUP-CAND** | No tracked reference found; appears to be interim/smoke/debug/superseded debris. Candidate to delete or physically archive later. |
| **INFRA** | Base model weights / dependencies. Keep. |

---

## Checkpoints (~7.9G)

| Path (under `checkpoints/`) | Size | Status | Note |
|---|---|---|---|
| `vlm_planner_lora_v8_20260604_v3` | 367M | **CURRENT** | Counterfactual Planner (remove) baseline; default adapter in `e2w_remove.py` |
| `vlm_planner_lora_add_v1_20260615` | ~370M | **CURRENT (bootstrap only)** | Add planner, narrow bootstrap SFT on self-insertion-inversion data; fits the contract-template distribution. **Eval is degenerate** (12 objects overlap train; each object's label JSON is verbatim-identical across backgrounds; placement/prompt fixed per object), so "24/24 valid" = format + 12-way recall, NOT generalization. Evidence: TRAINING only. Usable for an INTERFACE smoke; not reliable evidence of general add-planner capability. Pass via `e2w_add.py --planner-adapter` |
| `v04_real_overfit_14b_specfix_selfinsert_20260612` | 260M | **CURRENT** | Corrected v04 control branch (TRAINING evidence) |
| `v04_real_overfit_14b_specfix_selfinsert_20260612_pilot20` | 173M | **CURRENT** | 20-step pilot of the above (cited STATUS/ledger) |
| `vlm_planner_lora_physics_iq_v5_split_eval` | 627M | **CODE-DEFAULT** | Default in `tools/e2w_v0_common.py`; pre-v8 planner LoRA |
| `v04_real_overfit_14b_20260604` | 1.4G | **STALE-DOC** | Predates conditioning/Q3 fixes; "cannot prove corrected format" (STATUS) |
| `vlm_planner_lora_v8_20260604` | 367M | **ARCHIVED-REF** | Superseded by `_v3`; cited in archived handoff |
| `vlm_planner_lora_physics_iq_v7_targetfree_final_20260604` | 367M | **ARCHIVED-REF** | "Archived Planner References" (STATUS) |
| `vlm_planner_lora_physics_iq_v6_executable*` (×3) | 367M ea | **ARCHIVED-REF** | v6 executable-planner era; cited in archived handoff |
| `v04_overfit_phase1a_20260604` | 1.9M | **ARCHIVED-REF** | Cited in archived handoff |
| `vlm_planner_lora_physics_iq_v1..v4` (×4) | 627M ea (~2.5G) | **CLEANUP-CAND** | Pre-v8 planner LoRAs; no tracked reference |
| `vlm_planner_lora_v8_20260604_v2` | 367M | **CLEANUP-CAND** | Superseded by `_v3`; no current reference |
| `vlm_planner_lora_physics_iq_smoke` | 367M | **CLEANUP-CAND** | Smoke/toy checkpoint |
| `v04_real_overfit_14b_20260609_1f17fixed` | 173M | **CLEANUP-CAND** | Intermediate 14B run, no tracked reference |
| `v04_overfit_phase1a_debug_*` (gate05/gate025/tmp) | ~3.5M | **CLEANUP-CAND** | Debug overfit checkpoints |
| `Qwen2.5-VL-7B-Instruct`, `Wan2.1-VACE-14B`, `sam2`, `Qwen-Image-Edit` | 0–4K | **INFRA** | Base-model symlinks; keep |

---

## Runs (~7.9G)

| Path (under `runs/`) | Size | Status | Note |
|---|---|---|---|
| `counterfactual_bridge_skipvace_30_20260609T_run` | 1.5G | **CURRENT** | 30-sample remove-side STRUCTURAL gate |
| `physics_iq_for_simple_eval_20260613_fix_conditioning` | 276M | **CURRENT** | Post-conditioning-fix simple-eval rerun |
| `counterfactual_bridge_vace_interface_1_gpu2_20260609T_run` | 75M | **CURRENT** | Remove-side VACE INTERFACE smoke |
| `add_pipeline_interface_add_bg_000001_v2_20260609T152335Z` | 19M | **CURRENT** | First contract-safe add INTERFACE smoke |
| `e2w_v0_physics_iq` | 530M | **CODE-DEFAULT** | Default in `e2w_v0_common.py` & `physics_iq_for_simple_eval.py` (OLD_RUN_DIR) |
| `e2w_v0_1_physics_iq_teacher_grounded_eval` | 2.2G | **CODE-DEFAULT** | Default in `export_teacher_grounded_bundle.py`; teacher-grounded v0.1 era |
| `physics_iq_for_simple_eval` | 370M | **INVALIDATED** | Conditioning bug (future frames from source); also the *default output root* in `physics_iq_for_simple_eval.py:30` — see footgun below |
| `e2w_v0_2_full_cuda_20260602T0720Z` | 617M | **ARCHIVED-REF** | "Archived Planner References" (STATUS) |
| `e2w_v0_3_full_add_0076_pipeline_20260602T121720Z` | 67M | **ARCHIVED-REF** | "important historical run" (ledger); teacher/manual artifacts |
| `e2w_v6_*` / `e2w_v7_targetfree_final_*` planner runs (eval30 / remove8 / base / prompt variants) | 3.4M–14M ea | **ARCHIVED-REF** | v6/v7 executable-planner era; cited STATUS + archived handoff |
| `audit_gold_train_v6`, `audit_gold_eval_v6` | 13M / 1.5M | **ARCHIVED-REF** | v6 teacher-gold audit |
| `e2w_A6_*`, `e2w_A7_*` self-correction (manual) | 80K–224K | **ARCHIVED-REF** | Cited in archived A6 self-correction doc |
| `e2w_v0_3_quad_vace_add_0076_*` | ~35M | **ARCHIVED-REF** | Referenced by prefix in archived CONTRACT / v0.3 status |
| `add_pipeline_interface_add_bg_000001_20260609T024340Z` | 18M | **STALE-DOC** | Superseded add v1 (remove-residue prompt gap); cited STATUS/ledger/README |
| `e2w_v0_2b_qwen_vace_smoke_promptfix` | 620M | **CLEANUP-CAND** | v0.2 smoke; no tracked reference |
| `e2w_v0_2_contract_dry_codex` | 589M | **CLEANUP-CAND** | v0.2 dry run |
| `counterfactual_bridge_skipvace_5_*` (×2) | 212M ea | **CLEANUP-CAND** | Interim 5-sample bridge runs (superseded by the 30 gate) |
| `counterfactual_bridge_vace_interface_1_*` (offload / plain) | 36M ea | **CLEANUP-CAND** | Interim INTERFACE runs (superseded by `_gpu2_`) |
| `e2w_v8_pipeline_debug` | 292M | **CLEANUP-CAND** | Debug run (only in archived handoff) |
| `e2w_v0_2c_0077_promptfix_no_people` | 76M | **CLEANUP-CAND** | v0.2 variant |
| `e2w_v0_3_full_add_0076_{fresh_,}20260602T10*` (×2) | 32–34M | **CLEANUP-CAND** | v0.3 add variants |
| `e2w_v0_2_quad_vace_add_0076_20260602T093723Z` | 31M | **CLEANUP-CAND** | v0.2 add variant |
| `add_pipeline_interface_add_bg_000001_20260609T023256Z` | 18M | **CLEANUP-CAND** | Interim add attempt |
| `e2w_v0_2_full_rerun_20260602T0710Z` | 17M | **CLEANUP-CAND** | v0.2 rerun |
| `e2w_vlm_retrain_eval_20260603T125435Z` | 14M | **CLEANUP-CAND** | Retrain eval; no tracked reference |
| earlier `add_pipeline_interface_*` (06-08) + `e2w_v0_2_prompt*` checks + `*.log` | <1M ea | **CLEANUP-CAND** | Small interim/smoke runs and stray logs |

---

## Cleanup candidates summary

Roughly **~6G** of `CLEANUP-CAND` items have no tracked code/doc reference (largest: the four pre-v8
`vlm_planner_lora_physics_iq_v1..v4` ≈ 2.5G, and the `e2w_v0_2*` smoke/dry runs ≈ 1.8G). A further
~7G is referenced but historical (`STALE-DOC` / `ARCHIVED-REF`) or wired as a `CODE-DEFAULT`, so it is
**not** free to remove without also editing tracked code/docs.

Before deleting or physically archiving any `CLEANUP-CAND` item, re-verify it is unreferenced
(including in `.worktree/` branches):

```bash
grep -rnw <basename> $(git ls-files '*.py' '*.md') ; grep -rnw <basename> .worktree/
```

## Known footgun (separate from this index)

`tools/physics_iq_for_simple_eval.py:30` sets `DEFAULT_RUN_ROOT = runs/physics_iq_for_simple_eval` —
the run marked **INVALIDATED** above. The valid run is `..._20260613_fix_conditioning`. A default-args
run would write back into the invalidated-named directory. Consider repointing the default (separate
fix, not part of this index).
