from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def load_label_map(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    label2id = json.loads(path.read_text(encoding="utf-8"))
    id2label = {v: k for k, v in label2id.items()}
    return id2label, label2id


def build_dataset(data_dir: Path) -> DatasetDict:
    data_files = {
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "val.jsonl"),
        "test": str(data_dir / "test.jsonl"),
    }
    return load_dataset("json", data_files=data_files)


def align_labels_with_tokens(examples, tokenizer, label2id):
    max_length = 512
    encoded_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for tokens, tags in zip(examples["tokens"], examples["ner_tags"]):
        # Expand word-level BIO tags to subword-level tags.
        piece_ids: list[int] = []
        piece_labels: list[int] = []
        for word, tag in zip(tokens, tags):
            sub_ids = tokenizer.encode(word, add_special_tokens=False)
            if not sub_ids:
                sub_ids = [tokenizer.unk_token_id]

            first_label = label2id.get(tag, label2id["O"])
            if tag.startswith("B-"):
                cont_tag = "I-" + tag[2:]
            else:
                cont_tag = tag
            cont_label = label2id.get(cont_tag, label2id["O"])

            piece_ids.extend(sub_ids)
            piece_labels.append(first_label)
            piece_labels.extend([cont_label] * (len(sub_ids) - 1))

        input_ids = tokenizer.build_inputs_with_special_tokens(piece_ids)
        special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

        labels = []
        ptr = 0
        for is_special in special_mask:
            if is_special:
                labels.append(-100)
            else:
                labels.append(piece_labels[ptr] if ptr < len(piece_labels) else label2id["O"])
                ptr += 1

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        attention_mask = [1] * len(input_ids)
        encoded_batch["input_ids"].append(input_ids)
        encoded_batch["attention_mask"].append(attention_mask)
        encoded_batch["labels"].append(labels)

    return encoded_batch


def build_metrics_fn(id2label):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_predictions = []
        true_labels = []
        for pred, lab in zip(predictions, labels):
            cur_preds = []
            cur_labels = []
            for p, l in zip(pred, lab):
                if l == -100:
                    continue
                cur_preds.append(id2label[int(p)])
                cur_labels.append(id2label[int(l)])
            true_predictions.append(cur_preds)
            true_labels.append(cur_labels)
        result = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"],
            "accuracy": result["overall_accuracy"],
        }

    return compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT for token classification on BIO JSONL.")
    parser.add_argument("--data-dir", required=True, help="Directory containing train/val/test JSONL")
    parser.add_argument("--label-map", required=True, help="Path to label_map.json")
    parser.add_argument("--output-dir", required=True, help="Model output directory")
    parser.add_argument("--model-name", default="vinai/phobert-base-v2")
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    id2label, label2id = load_label_map(Path(args.label_map))
    ds = build_dataset(data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenized = ds.map(
        lambda x: align_labels_with_tokens(x, tokenizer, label2id),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_metrics_fn(id2label),
    )

    trainer.train()
    val_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"])
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    (output_dir / "metrics_val.json").write_text(
        json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics_test.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Training completed.")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
