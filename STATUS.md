# E2W 当前状态

最后更新：2026-06-16T00:00:00Z UTC。

本文件是当前项目状态真源。`docs/STATUS.md` 只保留指向本文件的摘要；不要把历史 handoff、README 示例或旧 run 当作当前状态。

## 总结

2026-06-16 更新（feat/unify-pipeline-interfaces 分支）：

- 接口统一为三个：`tools/e2w_remove.py` / `e2w_add.py` / `e2w_add_then_remove.py`，共享 `tools/e2w_pipeline_core.py`；add 新路径已弃用 v6/eval_vlm_planner，旧入口 `run_counterfactual_planner_pipeline.py` / `run_add_pipeline_interface.py` / `run_add_then_remove_pipeline.py` 只保留为 shim。
- add planner LoRA v1（`vlm_planner_lora_add_v1_20260615`）：self-insertion 反演 bootstrap SFT。证据等级 **仅 TRAINING**——eval 退化（train/eval 12 物体重叠且每物体标签逐字相同），不证明泛化。
- add 一等公民设计：first-frame edit 改为 **planner-bbox 约束的 masked inpaint**（物体只在 planner 选定区域生成），SAM2 同点 grounding + 一致性守卫；e2w_add 端到端 INTERFACE smoke 通过（overlap 0.796，产出 edited_video）。
- 2026-06-16 review 修复已落地：remove schema/prompt/VACE 失败会 fail loudly；add planner 要求 `primary_point` 落在 `primary_bbox` 内；masked-inpaint metadata 真实记录 mask 被消费；add/remove metadata 记录 alignment；add-then-remove evidence level 固定为合法的 `INTERFACE`，另用 `visual_candidate=true` 表示人工评审候选。
- 固定跑法见 `docs/UNIFIED_PIPELINE_RUNBOOK.md`。**诚实边界**：统一接口完成的是 current-spec INTERFACE/STRUCTURAL 对齐，不是 CONTROL/VISUAL/RESEARCH 成功；要让 VACE 真正吃准 quadmask 仍需要 control-branch 接入后的扰动测试和更大规模训练/评审。

当前已有一条 Counterfactual Planner -> grounding bridge -> current VACE runtime 的 remove-side INTERFACE smoke。仍不能声称 control / visual / research 成功。

- 当前正确 planner 设计是 Counterfactual Planner；兼容 schema id 为 `e2w.planner_output.v8_tool_augmented_grounding.v1`。
- Counterfactual Planner 目标是先生成 target-free counterfactual state，再由 grounding bridge 生成 `quadmask_npy` 并接入 current VACE runtime。
- archived executable-planner routes 已退出当前主线，仓库内材料统一归档，只作为历史 artifact/reference，不再作为现状基准。
- VACE Phase 1A control-branch 数据路线与 planner 训练路线独立，训练阶段不互相依赖。

在 control / visual 验收完成前，不要继续 package/report 或声称模型已学会可控 counterfactual editing。

2026-06-13 更新：首轮 Physics-IQ simple eval 的 remove 产物作废，不可用于人工视觉判断。原因是 remove runner 构造 `vace_conditioning_video` 时只替换了第 0 帧，后续帧仍来自 factual source video，导致 VACE 在第 1 帧以后重新看见本该去除的物体。当前 contract 已明确要求 `vace_conditioning_video` 为 edited first frame + zero-filled future frames；代码和 validator 已修复，需重跑 benchmark。

## 线 A - Planner 和 Grounding Bridge

### 当前目标

当前 Counterfactual Planner contract 来自 `docs/E2W_SPEC.md`：

```text
schema: e2w.planner_output.v8_tool_augmented_grounding.v1
required fields:
  target_ref
  edit_type
  counterfactual_state
  if_removed
```

Counterfactual Planner 不直接输出 `quadmask_spec`。正确 pipeline 是：

```text
original video + user remove request
-> Counterfactual Planner JSON
-> target_ref grounding with GroundingDINO/SAM2 or equivalent
-> quadmask_npy
-> vace_prompt from counterfactual_state / if_removed
-> first-frame-edited vace_conditioning_video
-> current six-input VACE runtime
```

### Counterfactual Planner 当前证据

最新 Counterfactual Planner seed_v3 run:

- checkpoint: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3`
- data: `/data/cwx/E2W/data/line_c_annotations/seed_v3.jsonl`
- split: train 136, eval 34
- train: 300 steps, epoch 8.8235
- train loss: 1.33311
- eval loss: 1.23665
- eval output: `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3/eval_v8_outputs.jsonl`
- eval summary: 34/34 JSON parse, 33/34 schema valid, 33/34 `if_removed` pass
- valid fill_type distribution: `background_continuation` 27, `occlusion_reveal` 6
- details: `docs/archive/experiments/A12_counterfactual_planner_seed_v3_20260608.md`

Evidence level: STRUCTURAL for planner text/schema compliance only.

### 当前 bridge / runtime 证据

Counterfactual Planner 已完成 remove-side 30 eval structural gate 和一条 VACE INTERFACE smoke：

- structural gate: `/data/cwx/E2W/runs/counterfactual_bridge_skipvace_30_20260609T_run`
  - 30/30 sampled eval rows status OK；
  - planner JSON parse/schema OK；
  - GroundingDINO bbox + SAM2 propagation OK；
  - all `quadmask_npy` value sets were `[0,127,255]` with nonzero Q0 and Q2；
  - `generation_mask` values `[255]`, `generation_mask_is_full_domain=true`；
  - `vace_prompt_valid=true`；
  - `source_video_passed_to_vace=false`；
  - bbox confidence range: `0.3583865463733673` to `0.9003759026527405`。
- VACE INTERFACE smoke: `/data/cwx/E2W/runs/counterfactual_bridge_vace_interface_1_gpu2_20260609T_run`
  - sample `4fe6619a47`, target_ref `a bathroom sink`；
  - first-frame edit OK；
  - VACE backend returncode `0`；
  - output video: `/data/cwx/E2W/runs/counterfactual_bridge_vace_interface_1_gpu2_20260609T_run/edited_video_4fe6619a47.mp4`；
  - VACE runtime inputs recorded with current E2W names: `vace_conditioning_video`, `quadmask_npy`, `generation_mask`, `operation`, `vace_prompt`, `frame_num`。

Evidence level:

- bridge: STRUCTURAL on 30 eval samples；
- remove full path: INTERFACE on 1 sample；
- visual quality, operation control, quadmask control, and learned VACE semantics: not established。

Code-side fixes made during this gate:

- unified remove runner (`tools/e2w_remove.py`; legacy `run_counterfactual_planner_pipeline.py` is a shim) preserves SAM2 primary as Q0 instead of collapsing all primary pixels into Q1；
- first-frame Qwen Image Edit uses model CPU offload when available to avoid single-GPU OOM；
- regression coverage added in `tests/test_counterfactual_planner_bridge.py`。

## Archived Planner References

以下 archived executable-planner artifacts 只用于历史对比，不再作为当前 planner 基准；repo 内相关脚本、测试、fixtures 已移入 `tools/archive/` 和 `tests/archive/`：

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

已核查状态（2026-06-12）：

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
- 修正后真实 14B self-insertion overfit 已完成：
  - checkpoint: `/data/cwx/E2W/checkpoints/v04_real_overfit_14b_specfix_selfinsert_20260612`
  - pilot: `/data/cwx/E2W/checkpoints/v04_real_overfit_14b_specfix_selfinsert_20260612_pilot20`
  - steps: 200，metrics rows: 201
  - `real_weights_loaded=true`, `mock_or_toy=false`
  - `conditioning_inputs=["edited_first_frame","quadmask","vace_prompt"]`
  - `source_frames_used_for_vace_conditioning=false`
  - `generation_mask_is_full_domain=true`, `generation_mask_values=[255]`, `generation_mask_encodes_quadmask_semantics=false`
  - `q3_loss_weight=0.1`, final `gate=0.046739354729652405`
  - `metrics.jsonl` records `diffusion_loss`, `q3_mse`, `q3_loss_weight`, `grad_norm`, `gate`, and `quadmask_alignment`
  - Evidence level: TRAINING only.
- 已验证 branch tests: `tests.test_v04_anchor_manifest_audit`, `tests.test_v04_control_branch_freeze`, and `tests.test_v04_control_branch_gradients` 共 19 tests OK。

仍未完成：

- 将 trained v04 control branch checkpoint 接入 VACE inference runner。
- operation swap / Q0 perturbation / Q2 perturbation / Q3 preservation 验收。
- Phase 1B Kubric/HUMOTO 真实物理 counterfactual pairs。

线 A 和线 B 只在推理时汇合：

```text
用户指令 -> Counterfactual Planner -> grounding bridge -> quadmask + VACE prompt -> VACE -> edited video
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

Code-side follow-up on 2026-06-09:

- add runner 不再用 archived v6 executable planner prompt builder；
- `eval_vlm_planner.py` 新增 `parse_add_planner_json()` 回退路径，operation=add 时绕过 v6 schema 必填字段检查；
- add planner split prompt 改为 current add contract，要求 model-produced `vace_prompt`、top-level `target_ref`、正向 add wording、point/bbox grounding；
- metadata 改为记录 top-level raw `target_ref`，并记录 `vace_conditioning_video` future frames 为 zero-filled placeholders。

Clean add INTERFACE run after code fix (2026-06-09):

```text
/data/cwx/E2W/runs/add_pipeline_interface_add_bg_000001_v2_20260609T152335Z
```

- `add_contract_parse_fallback_used = true`（新 parser 路径已触发）
- `vace_prompt_source = planner_model`，`manual_or_teacher_vace_prompt_used = false`
- actual `vace_prompt`：`A cozy dining room scene with a red mug placed on the table near the center, casting a warm glow in the dimly lit room.`（正向 add wording，无 remove-residue ✅）
- all `success_criteria` pass；`visual_quality_evaluated = false`

This run is the first contract-safe add INTERFACE smoke. Visual quality and learned VACE add semantics are not established.

## 当前阻塞点

主 blocker 已从接口/bridge structural alignment 前移到 control / visual / training evidence：统一接口现在按 current spec 固定为 `e2w_remove.py` / `e2w_add.py` / `e2w_add_then_remove.py`，但这仍不证明 learned control。

旧 `/data/cwx/E2W/runs/physics_iq_for_simple_eval` 不能作为 remove VISUAL candidate，因为 remove conditioning future frames 混入了 source video。必须使用修复后的新 run root 重跑。

当前不要从 archived executable-planner checkpoint 继续跑 full forward pass。

## Next Actions

- [x] 1. 修正 remove bridge contract（当前入口 `tools/e2w_remove.py`；旧 `tools/run_counterfactual_planner_pipeline.py` 为 shim），使输出符合 `docs/E2W_SPEC.md`：
  - `generation_mask` full-domain all-255；
  - metadata 使用 E2W-level 六输入命名；
  - backend `src_video`/`prompt` 只作为 adapter 内部名；
  - 记录 planner JSON -> grounding -> quadmask -> vace_prompt -> VACE inputs 的证据链。
- [x] 2. 对 Counterfactual Planner eval 做小规模 bridge smoke，先 `--skip-vace` 验证 STRUCTURAL：
  - planner JSON parse/schema；
  - target_ref grounding 成功；
  - quadmask shape/value；
  - full-domain generation_mask；
  - vace_prompt target-free。
- [x] 3. 通过 structural gate 后，再跑一条 VACE INTERFACE smoke。
- [x] 4. 扩大到 30 eval structural gate，确认 bridge 稳定性。
- [x] 5a. 实现 CONTROL 验收工具 `tools/run_control_perturbation_test.py`（operation_swap / q0_suppressed / q0_shifted / q3_preservation 四项扰动）。
- [x] 5b. 完成修正后真实 14B self-insertion overfit，建立 TRAINING 证据。
- [ ] 5c. 将 trained v04 control branch checkpoint 接入 VACE inference runner，并用新 checkpoint 运行四项 CONTROL 扰动测试。
- [ ] 6. 人工或模型评审 remove + add 输出，达到 VISUAL 证据后再 package/report。
