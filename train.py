#!/usr/bin/env python

import argparse
from typing import Dict, List

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


MAIN_CHARS: List[str] = [
    "Joey Tribbiani",
    "Monica Geller",
    "Chandler Bing",
    "Phoebe Buffay",
    "Rachel Green",
    "Ross Geller",
]

# How many previous lines (within the same split) to use as context
CONTEXT_WINDOW = 2


def build_datasets_from_csv(csv_path: str) -> DatasetDict:
    """
    Load the Friends CSV and convert rows into HF datasets with a 'messages' field
    suitable for Granite chat SFT, including up to CONTEXT_WINDOW previous lines
    of dialogue (within the same split) as context.
    """
    # Load the full CSV
    raw = load_dataset("csv", data_files={"full": csv_path})["full"]

    # Keep only the six main characters
    raw = raw.filter(lambda ex: ex["speaker"] in MAIN_CHARS)

    # Build split-wise datasets based on the 'split' column
    unique_splits = sorted(set(raw["split"]))
    split_datasets: Dict[str, object] = {}

    for name in unique_splits:
        split_ds = raw.filter(lambda ex, n=name: ex["split"] == n)

        # Pre-extract columns so we can index them inside map
        speakers_col = split_ds["speaker"]
        texts_col = split_ds["text"]
        n_rows = len(split_ds)

        def add_messages_with_context(example, idx):
            target_line = example["text"]
            target_speaker = example["speaker"]

            # Collect up to CONTEXT_WINDOW previous lines in this split
            # ----- previous lines -----
            prev_start = max(0, idx - CONTEXT_WINDOW)
            prev_lines = []
            for j in range(prev_start, idx):
                prev_speaker = speakers_col[j]
                prev_text = texts_col[j]
                prev_lines.append(f"{prev_speaker}: {prev_text}")

            # ----- succeeding lines -----
            next_end = min(n_rows, idx + CONTEXT_WINDOW + 1)
            next_lines = []
            for j in range(idx + 1, next_end):
                next_speaker = speakers_col[j]
                next_text = texts_col[j]
                next_lines.append(f"{next_speaker}: {next_text}")

            # Build a single context string
            if prev_lines or next_lines:
                context_sections = []
                if prev_lines:
                    context_sections.append(
                        "Previous lines:\n" + "\n".join(prev_lines)
                    )
                if next_lines:
                    context_sections.append(
                        "Following lines:\n" + "\n".join(next_lines)
                    )
                context_block = "\n\n".join(context_sections)
            else:
                context_block = "None."

            user_content = (
                'You are an expert annotator of the TV show "Friends".\n\n'
                "Below is a snippet of dialogue from the show. Use the surrounding "
                "context and the target line to identify which character says the "
                "target line.\n\n"
                "Context (previous and following lines):\n"
                f"{context_block}\n\n"
                f'Target line: "{target_line}"\n\n'
                "Question: Which character says the target line?\n"
                "Answer with exactly one name from:\n"
                + ", ".join(MAIN_CHARS)
                + "."
            )

            example["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "You label dialogue lines from Friends with the correct "
                        "character name. Only answer with a single character name."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": target_speaker,
                },
            ]
            return example


        # Add 'messages' with context to each row in this split
        split_ds = split_ds.map(add_messages_with_context, with_indices=True)
        split_datasets[name] = split_ds

    return DatasetDict(split_datasets)


def get_tokenizer_and_model(model_name: str):
    # Try fast tokenizer first, fall back to slow if cache / Rust tokenizer is broken
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print("Fast tokenizer failed, falling back to slow tokenizer. Error was:")
        print(e)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model



def main():
    parser = argparse.ArgumentParser(
        description="SFT Granite-4.0-micro on Friends speaker classification."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Friends CSV (columns: speaker,text,split).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ibm-granite/granite-4.0-micro",
        help="Base Granite model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./granite-friends-speaker-lora",
        help="Where to save fine-tuned model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Max sequence length for SFT.",
    )
    # --- wandb-related args ---
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="If set, log training to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cse595-friends-granite",
        help="wandb project name (if --use_wandb).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional wandb run name (if --use_wandb).",
    )

    args = parser.parse_args()

    print("Loading datasets from:", args.data_path)
    dataset = build_datasets_from_csv(args.data_path)

    has_train = "train" in dataset
    if not has_train:
        raise ValueError(
            "No 'train' split found in CSV. Make sure the 'split' column "
            "contains at least some rows with value 'train'."
        )

    train_dataset = dataset["train"]
    eval_dataset = dataset["val"] if "val" in dataset else None

    print("Loading tokenizer and model:", args.model_name)
    tokenizer, model = get_tokenizer_and_model(args.model_name)

    def formatting_func(example):
        # example["messages"] is a list of {role, content}
        # We return a single string per example, using the chat template.
        return tokenizer.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False,
        )

    # LoRA config (QLoRA-style SFT)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # wandb integration:
    report_to = []
    if args.use_wandb:
        if wandb is None:
            raise ImportError(
                "wandb is not installed, but --use_wandb was set. "
                "Install with `pip install wandb`."
            )
        report_to = ["wandb"]
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,           # how often to log (train_loss, etc.)
        save_strategy="epoch",      # save at each epoch
        fp16=False,
        report_to=report_to,        # [] or ["wandb"]
        load_best_model_at_end=False,
        # max_seq_length=args.max_seq_len,
    )

    print("Starting SFT training...")
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        peft_config=peft_config,
        # max_seq_length=args.max_seq_len,
        args=training_args
    )

    trainer.train()
    print("Saving model to:", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.use_wandb:
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
