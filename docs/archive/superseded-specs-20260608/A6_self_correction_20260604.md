# 实验 A6：同模型改写任务 prompt 验证 target-free 自修正能力

## 实验动机

remove8 gate 在 planner SFT 重训后只有 1/8 通过。诊断显示，失败主因不是 gold label 污染，而是模型推理时在 `counterfactual_expectation.if_removed` 里重新生成了 target 名称、target 子词或材料词，导致 `serialize_vace_prompt()` 的 VACE prompt contract hard fail。

本实验不修改模型、不修改 validator，只用同一个 LoRA checkpoint 给模型一个显式“改写 if_removed”任务，验证模型在明确禁止词约束下是否能生成 target-free 的反事实场景描述。

## 方法说明

- 上次 remove8 推理 run：`/data/cwx/E2W/runs/e2w_v7_targetfree_final_planner_remove8_20260604T0904Z`
- 本实验输出 run：`/data/cwx/E2W/runs/e2w_A6_self_correction_20260604TmanualZ`
- LoRA checkpoint：`/data/cwx/E2W/checkpoints/vlm_planner_lora_physics_iq_v7_targetfree_final_20260604`
- Base model：`/data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct`
- 生成设置：text-only prompt，`do_sample=False`，`max_new_tokens=160`

实现步骤：

1. 读取 `tools/eval_vlm_planner.py`，沿用其模型加载方式：`AutoProcessor` + `Qwen2_5_VLForConditionalGeneration` + `PeftModel.from_pretrained()`。
2. 读取 `tools/e2w_v0_common.py`，使用 `_target_term_variants()` 和 `serialize_vace_prompt()`。
3. 对 7 个失败样本 `0052 0056 0070 0076 0112 0128 0341`：
   - 读取上次推理的完整 `planner_pred/<row_id>/raw.pred.json`。
   - 将原 JSON 中 `counterfactual_expectation.if_removed` 替换为 `[NEEDS REWRITE]` 后贴入 prompt。
   - 按 `serialize_vace_prompt()` 同一路径从 `edit_plan.pred.json` 生成 forbidden word list。
   - 要求模型只输出新的 `if_removed` 文本，不输出 JSON/字段名/解释。
   - 将新文本写回 `edited_scene.caption`、`edited_scene.outcome_effects`、`operation_details.physical_consequences`，调用 `serialize_vace_prompt()` 验证。

## 逐行结果

| row_id | 原 if_removed（前50字） | 新 if_removed（前50字） | pass/fail | 违规词 |
|---|---|---|---|---|
| 0052 | The Newton's cradle will be without any hanging me… | The Newton's cradle frame and scissors remain unch… | pass | — |
| 0056 | The stick is no longer attached to the black rotat… | The black rotating platform remains stationary wi… | pass | — |
| 0070 | The clear glass with water inside is no longer pre… | The table surface remains smooth and unobstructed… | fail | `glass`, `water` |
| 0076 | The yellow mug is no longer present on the wooden … | The wooden table remains unchanged, with its surf… | pass | — |
| 0112 | The black balloon is no longer present on the tabl… | The table remains clear and unobstructed, with no… | fail | `black balloon`, `balloon` |
| 0128 | The shallow dish of light blue liquid is no longer… | The wooden table remains unchanged, with its surf… | fail | `shallow dish of light blue liquid`, `light blue liquid`, `shallow dish`, `blue liquid`, `liquid`, `dish` |
| 0341 | The tall glass containing blue liquid is no longer… | The center of the frame is now empty, with no tal… | fail | `tall glass containing blue liquid`, `blue liquid`, `tall glass`, `liquid`, `glass` |

通过率：3/7。

## 结论

- 模型展示了一定 target-free 推理能力：在 0052、0056、0076 上，面对显式 forbidden word list 和“只改写 if_removed”的任务，模型能生成通过 `serialize_vace_prompt()` 的 target-free 文本。
- 但该能力不稳定：0070、0112、0128、0341 仍然在明确约束下复述 target 名称、target 子词或材料词，并且有样本继续使用 `absence`、`removal`、`present/disappeared` 这类“某物不在了”的表述。
- 因为 3 个样本能通过，同模型并非完全缺失 target-free 描述能力；更倾向于说明当前 SFT 训练量/训练分布/指令约束覆盖不足，模型尚未稳定内化 VACE-facing target-free contract，而不是能力完全缺失。


## A7 优化 prompt 结果

### 方法

A7 在 A6 自改写 prompt 的 `[要求]` 部分增加了“只描述当前存在的物体和状态、禁止负向陈述”的约束：禁止 `no X`、`without X`、`X is absent`、`no longer`、`empty of` 等句式，并要求模型把场景想象成一张从始至终没有目标物体、也不知道目标物体存在的照片来描述。

为保持和 A6 的英文 VACE 文本可比性，本次重跑保留了 `Use English only.` 约束。实验 artifacts：`/data/cwx/E2W/runs/e2w_A7_self_correction_negative_free_english_20260604TmanualZ`。

### 0112 / 0341 / 0070 重跑结果

| row_id | 完整新 if_removed | pass/fail | 违规词 |
|---|---|---|---|
| 0112 | The table remains unchanged, with no black balloon present. | fail | `black balloon`, `balloon` |
| 0341 | The scene remains unchanged, with no tall glass containing blue liquid present. | fail | `tall glass containing blue liquid`, `blue liquid`, `tall glass`, `liquid`, `glass` |
| 0070 | The table on the right side now has no glass of water, but the rest of the scene remains unchanged. | fail | `glass of water`, `glass`, `water` |

结果：3 个样本仍然全部失败。优化 prompt 明确禁止 `no X` 和其它负向句式后，模型仍然输出了 `with no black balloon present`、`with no tall glass containing blue liquid present`、`has no glass of water` 这类负向目标陈述，说明单靠推理时指令仍不能稳定压住这一模式。

### 0128 validator 分析

A6 中 0128 的完整改写输出：

> The wooden table remains unchanged, with its surface and surrounding area unaffected by the removal of the shallow dish of light blue liquid.

`_target_term_variants()` 为 0128 生成的完整 forbidden list：

```json
["shallow dish of light blue liquid", "shallow dish", "dish", "light blue liquid", "blue liquid", "liquid"]
```

判断：0128 不是 validator 过严导致的误报。虽然 forbidden list 中确实包含 `dish`、`liquid` 这类较通用词，理论上在其它样本中可能误伤与目标无关的通用表述；但 A6 的 0128 输出直接包含了 `the removal of the shallow dish of light blue liquid`，即完整目标短语和 removal wording。因此本例应判定为模型真的提到了目标物体，而不是 validator 对无关通用词的过严误报。
