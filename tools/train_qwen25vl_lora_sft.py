#!/usr/bin/env python3
"""LoRA SFT for Qwen2.5-VL planner JSON generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


class PlannerSFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        video_fps: float,
        min_pixels: int,
        max_pixels: int,
    ) -> None:
        self.rows = rows
        self.video_fps = video_fps
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        user_text = row["messages"][0]["content"]
        assistant_text = row["messages"][1]["content"]
        video_path = row["video"]
        if not Path(video_path).exists():
            raise FileNotFoundError(video_path)
        return {
            "id": row["id"],
            "prompt_messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": self.video_fps,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
            "assistant_text": assistant_text,
        }


class QwenVLCollator:
    def __init__(self, processor: AutoProcessor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError("This collator intentionally supports batch_size=1 only.")

        feature = features[0]
        prompt_messages = feature["prompt_messages"]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": feature["assistant_text"]}
        ]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            prompt_messages, return_video_kwargs=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        labels = full_inputs["input_ids"].clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        labels[full_inputs["attention_mask"] == 0] = -100
        full_inputs["labels"] = labels
        return full_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--min-pixels", type=int, default=50176)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("--batch-size must be 1 for this video collator.")

    rows = load_jsonl(args.train_jsonl, args.max_samples)
    if not rows:
        raise ValueError(f"No examples found in {args.train_jsonl}")
    eval_rows = load_jsonl(args.eval_jsonl) if args.eval_jsonl else []

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    dataset = PlannerSFTDataset(
        rows,
        video_fps=args.video_fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    eval_dataset = None
    if eval_rows:
        eval_dataset = PlannerSFTDataset(
            eval_rows,
            video_fps=args.video_fps,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    collator = QwenVLCollator(processor)

    if args.dry_run:
        batch = collator([dataset[0]])
        printable = {
            key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
            for key, value in batch.items()
        }
        print(
            json.dumps(
                {
                    "train_examples": len(dataset),
                    "eval_examples": len(eval_dataset) if eval_dataset else 0,
                    "first_batch": printable,
                },
                indent=2,
            )
        )
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps or args.save_steps,
        per_device_eval_batch_size=1,
        save_total_limit=2,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train()
    if eval_dataset is not None:
        print(json.dumps(trainer.evaluate(), indent=2, sort_keys=True))
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
