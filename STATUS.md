# E2W 当前状态

最后更新：2026-06-09 UTC。

本文件是当前项目状态真源。`docs/STATUS.md` 只保留指向本文件的摘要；不要把历史 handoff、README 示例或旧 run 当作当前状态。

## 总结

当前不是可以直接 full forward pass 的状态。

- 当前正确 planner 设计是 v8：`e2w.planner_output.v8_tool_augmented_grounding.v1`。
- v8 目标是先生成 target-free counterfactual state，再由 grounding bridge 生成 `quadmask_npy` 并接入 current VACE runtime。
- v7 / v0.2 executable planner 已退出当前主线，只作为历史 artifact/reference，不再作为现状基准。
- VACE Phase 1A control-branch 数据路线与 planner 训练路线独立，训练阶段不互相依赖。

在 v8 planner -> grounding bridge -> current VACE runtime 的结构性对齐完成前，不要继续 package/report 或声称 full forward 成功。

## 线 A - Planner 和 Grounding Bridge

### 当前目标

当前 planner contract 来自 `docs/E2W_SPEC.md`：

```text
schema: e2w.planner_output.v8_tool_augmented_grounding.v1
required fields:
  target_ref
  edit_type
  counterfactual_state
  if_removed
```

v8 planner 不直接输出 `quadmask_spec`。正确 pipeline 是：

```text
original video + user remove request
-> v8 planner JSON
-> target_ref grounding with GroundingDINO/SAM2 or equivalent
-> quadmask_npy
-> vace_prompt from counterfactual_state / if_removed
-> first-frame-edited vace_conditioning_video
-> current six-input VACE runtime
```

### v8 当前证据

最新 v8 seed_v3 run:

- checkpoint: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3`
- data: `/data/cwx/E2W/data/line_c_annotations/seed_v3.jsonl`
- split: train 136, eval 34
- train: 300 steps, epoch 8.8235
- train loss: 1.33311
- eval loss: 1.23665
- eval output: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3/eval_v8_outputs.jsonl`
- eval summary: 34/34 JSON parse, 33/34 schema valid, 33/34 `if_removed` pass
- valid fill_type distribution: `background_continuation` 27, `occlusion_reveal` 6
- details: `docs/experiments/A12_seed_v3_v8_v3_20260608.md`

Evidence level: STRUCTURAL for planner text/schema compliance only.

### 当前缺口

v8 尚未完成 current runtime 对齐：

- grounding bridge 还没有被验收为 current spec 的结构性基准；
- `target_ref -> quadmask_npy` 需要稳定审计；
- `generation_mask` 必须改为 current spec 的 full-domain all-255，而不是从 quadmask 派生局部 mask；
- VACE 层 metadata 必须记录 E2W-level 名称：`vace_conditioning_video`, `vace_prompt`, `quadmask_npy`, `generation_mask`, `operation`, `frame_num`；
- 不能把 backend adapter 的 `src_video` / `prompt` 名称当作当前 E2W runtime contract；
- 还没有 v8 full bridge 的 30 eval / smoke structural gate。

## Archived Planner References

以下 v6/v7/v0.2 artifacts 只用于历史对比，不再作为当前 planner 基准：

- `/data/cwx/E2W/runs/e2w_v7_targetfree_final_planner_remove8_20260604T0904Z`
- `/data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v7_targetfree_final_20260604`
- `/data/cwx/E2W/data/physics_iq_vlm_sft/vlm_planner_sft_eval_v7_targetfree_final.jsonl`
- `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_eval30_20260603T1029Z`
- `/data/cwx/E2W/runs/e2w_v6_final_rules_planner_remove8_20260603T1038Z`
- `/data/cwx/E2W/runs/e2w_v6_base_planner_remove8_20260603TbaseZ`
- `/data/cwx/E2W/runs/e2w_v0_2_full_cuda_20260602T0720Z`
- `docs/archive/superseded-specs-20260608/A6_self_correction_20260604.md`
- `docs/archive/superseded-specs-20260608/vlm_planner_handoff_20260603.md`

Do not use these to decide current next actions except for historical failure analysis.

## 线 B - VACE Phase 1A Control Branch

该路线独立于 planner SFT 训练，目标是训练 VACE 侧的 gated causal control branch。

已核查状态（2026-06-08）：

- worktree: `/home/cwx/E2W/.worktree/feat/phase1-v04-control-branch`
- branch head: `1f17ea4 按 E2W_SPEC 修正 VACE 训练元数据`
- B1 下载 80 条 Pexels raw videos: `8de58bb`
- B2 标准化 80 条候选背景 `[81,480,832,3]`: `4b234e7`
- B3 筛选 16 条 clean backgrounds，58/80 自动通过: `fc39ad0`
- B4 生成 SI-A/B/C composites，12 个，0 audit failure: `b121c51`
  - train manifest 16 行，8 add + 8 remove
  - eval manifest 8 行，4 add + 4 remove
- `overfit_16.jsonl` 已有 16 行，8 add + 8 remove，并包含 `edited_first_frame` 和 `vace_prompt` 字段。
- 训练脚本 `tools/train_v04_control_branch_real_overfit.py` 已改为：
  - 使用 `edited_first_frame` 构造 VACE conditioning；
  - 在训练脚本内合成 full-domain generation mask；
  - 使用 `vace_prompt`；
  - 加入 Q3 latent MSE loss。
- 历史真实 14B run `/data/cwx/E2W/checkpoints/v04_real_overfit_14b_20260604` 在这些修正之前产生，`final_gate = 0.022334493696689606`，不能证明修正后的训练格式有效。
- 已验证 branch tests: `tests.test_v04_anchor_manifest_audit`, `tests.test_v04_control_branch_freeze`, and `tests.test_v04_control_branch_gradients` 共 19 tests OK。

仍未完成：

- 按 1f17 修正后的格式重跑真实 14B overfit。
- operation swap / Q0 perturbation / Q2 perturbation / Q3 preservation 验收。
- Phase 1B Kubric/HUMOTO 真实物理 counterfactual pairs。

线 A 和线 B 只在推理时汇合：

```text
用户指令 -> v8 planner -> grounding bridge -> quadmask + VACE prompt -> VACE -> edited video
```

## 线 C - Add Pipeline INTERFACE Smoke

`feat/add-pipeline` adds a current-spec add pipeline smoke path. Verified run:

```text
/data/cwx/E2W/runs/add_pipeline_interface_add_bg_000001_20260609T024340Z
```

Evidence level: INTERFACE only.

Verified acceptance facts:

- Input data: `/data/cwx/E2W/data/phase1a_pexels_self_insert_v1/02_background_clean/videos_mp4/bg_000001.mp4`
- User prompt: `Add a red mug on the table near the center of the image.`
- `vace_prompt_source = planner_model`
- `manual_or_teacher_vace_prompt_used = false`
- `planner_output_manually_modified = false`
- `accepted_point_only_for_add_interface = true` because the planner produced valid primary point grounding but no bbox.
- `edited_first_frame.png`, `vace_conditioning_video.mp4`, `quadmask.npy`, `generation_mask.npy`, and `edited_video.mp4` exist.
- `quadmask.npy` shape `[21,480,832]`, dtype `uint8`, values `[0,127,255]`.
- `generation_mask.npy` shape `[21,480,832]`, dtype `uint8`, values `[255]`.
- `source_video_passed_to_vace = false` at the E2W runtime-contract level.
- `visual_quality_evaluated = false`.

Artifact audit on 2026-06-09 found a prompt-contract gap:

- actual `vace_prompt`: `The red mug is no longer present on the table.`
- this is remove-residue text under `operation=add`, violating the current add prompt rule in `docs/E2W_SPEC.md`.

Therefore this run remains INTERFACE/provenance smoke only. It does not prove contract-safe add prompting, visual quality, learned planner add quality, or learned VACE add semantics.

## 当前阻塞点

主 blocker 是 v8 已经成为正确 planner design，但后续 grounding bridge / runtime adapter 仍未按 current spec 收敛。

当前不要从 v6/v7 planner checkpoint 继续跑 full forward pass。

## Next Actions

- [ ] 1. 修正 `tools/run_e2w_pipeline_v8.py` 的 bridge contract，使输出符合 `docs/E2W_SPEC.md`：
  - `generation_mask` full-domain all-255；
  - metadata 使用 E2W-level 六输入命名；
  - backend `src_video`/`prompt` 只作为 adapter 内部名；
  - 记录 planner JSON -> grounding -> quadmask -> vace_prompt -> VACE inputs 的证据链。
- [ ] 2. 对 v8 eval 做小规模 bridge smoke，先 `--skip-vace` 验证 STRUCTURAL：
  - planner JSON parse/schema；
  - target_ref grounding 成功；
  - quadmask shape/value；
  - full-domain generation_mask；
  - vace_prompt target-free。
- [ ] 3. 通过 structural gate 后，再跑一条 VACE INTERFACE smoke。
- [ ] 4. 只有 v8 bridge structural gate 通过后，才进入 package/report/freshness audit。
