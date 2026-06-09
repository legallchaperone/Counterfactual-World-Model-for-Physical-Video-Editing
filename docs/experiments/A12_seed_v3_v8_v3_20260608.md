# A12: seed_v3 planner SFT v8 v3

Date: 2026-06-08 UTC

Scope: E2W line A, main branch. Retrain the v8 planner-text LoRA using `seed_v3.jsonl` with the same training configuration as `vlm_planner_lora_v8_20260604_v2`, then evaluate fill-type diversity on the 34-row eval split.

## Data

Source:

- `/data/cwx/E2W/data/line_c_annotations/seed_v3.jsonl`

Generated SFT split:

- `/data/cwx/E2W/data/planner_sft_v8_seed_v3/train.jsonl`
- `/data/cwx/E2W/data/planner_sft_v8_seed_v3/eval.jsonl`

Split counts:

- total: 170
- train: 136
- eval: 34

Source distribution:

- total: ryvos 120, davis 50
- train: ryvos 96, davis 40
- eval: ryvos 24, davis 10

## Training

Checkpoint:

- `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3`

Configuration:

- base model: `/data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct`
- max steps: 300
- batch size: 1
- gradient accumulation: 4
- learning rate: 1e-5
- LoRA r/alpha/dropout: 8/16/0.05
- save steps: 300
- logging steps: 1

Training summary:

- train runtime: 704.482 sec
- epoch: 8.8235
- train loss: 1.3331095055739084
- eval loss: 1.236649990081787

Log:

- `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3.train.log`

## Eval Inference

Generated outputs:

- `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3/eval_v8_outputs.jsonl`

Eval log:

- `/data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3/eval_v8.log`

Metrics:

- count: 34
- json_parse_ok: 34/34, 100.0%
- schema_valid: 33/34, 97.1%
- if_removed pass: 33/34, 97.1%

Fill type distribution over valid samples:

- `background_continuation`: 27
- `occlusion_reveal`: 6

Key result: v3 does produce a non-`background_continuation` fill type. `occlusion_reveal` appears in 6 valid eval outputs.

Failure:

- `ryvos/2b904b76c9`: JSON parsed, but schema validation failed because `serialize_vace_prompt()` reported target-contaminated planner text.

## Commands

```bash
cd /home/cwx/E2W

/data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/build_sft_dataset_v8.py \
  --input /data/cwx/E2W/data/line_c_annotations/seed_v3.jsonl \
  --output-dir /data/cwx/E2W/data/planner_sft_v8_seed_v3 \
  --seed 42 \
  --train-ratio 0.8

CUDA_VISIBLE_DEVICES=7 /data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/train_qwen25vl_lora_sft.py \
  --train-jsonl /data/cwx/E2W/data/planner_sft_v8_seed_v3/train.jsonl \
  --eval-jsonl /data/cwx/E2W/data/planner_sft_v8_seed_v3/eval.jsonl \
  --base-model /data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct \
  --output-dir /data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3 \
  --max-steps 300 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 1e-5 \
  --save-steps 300 \
  --logging-steps 1

CUDA_VISIBLE_DEVICES=7 /data/cwx/conda/envs/edit2world-phase1-real/bin/python tools/eval_planner_v8_sft.py \
  --eval-jsonl /data/cwx/E2W/data/planner_sft_v8_seed_v3/eval.jsonl \
  --base-model /data/cwx/Edit2World-unified/checkpoints/Qwen2.5-VL-7B-Instruct \
  --adapter /data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3 \
  --output-jsonl /data/cwx/E2W/checkpoints/vlm_planner_lora_v8_20260604_v3/eval_v8_outputs.jsonl \
  --max-new-tokens 768 \
  --temperature 0.0 \
  --sample-count 5
```
