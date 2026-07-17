from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def read_manifest_splits(path: Path) -> dict[str, str]:
    splits: dict[str, str] = {}
    if not path.exists():
        return splits
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        splits[str(row["case_id"])] = str(row.get("split", ""))
    return splits


def report_json(row: dict[str, Any]) -> str:
    finding_id = "LORA_REFERENCE_REPORT"
    value = {
        "clinical_history": row.get("indication") or "Not provided",
        "technique": "Chest radiographs.",
        "findings": [{"text": str(row.get("findings") or "No acute cardiopulmonary abnormality is identified."), "evidence_ids": [finding_id]}],
        "impression": [{"text": str(row.get("impression") or "No acute cardiopulmonary abnormality."), "evidence_ids": [finding_id]}],
        "recommendations": [],
        "uncertainties": [],
        "ai_disclaimer": "AI-generated draft report for physician review only.",
    }
    return json.dumps(value, ensure_ascii=False)


def prompt_for(row: dict[str, Any]) -> str:
    payload = {
        "study_uid": row.get("uid"),
        "modality": "DX",
        "body_part": "CHEST",
        "clinical_context": {"chief_complaint": row.get("indication") or None},
        "image_findings": [{
            "finding_id": "LORA_REFERENCE_REPORT",
            "finding_type": "training_reference",
            "location": "chest radiographs",
            "confidence": 1.0,
        }],
        "retrieved_evidence": [],
        "required_output_style": {
            "language": "English",
            "sections": ["FINDINGS", "IMPRESSION"],
            "instruction": "Use concise IU X-Ray style and return valid ReportDraft JSON.",
        },
    }
    return (
        "You are an attending radiologist writing an English IU X-Ray style chest radiograph report.\n"
        "Use only the supplied image findings, clinical context, and retrieved evidence.\n"
        "Return only valid JSON matching the ReportDraft schema.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}\n\nOUTPUT_JSON:\n"
    )


class JsonReportDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples = []
        for row in rows:
            prompt = prompt_for(row)
            target = report_json(row) + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full = tokenizer(prompt + target, add_special_tokens=False, truncation=True, max_length=max_length)
            input_ids = full.input_ids
            labels = input_ids.copy()
            cutoff = min(len(prompt_ids), len(labels))
            labels[:cutoff] = [-100] * cutoff
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": full.attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate(features: list[dict[str, Any]], tokenizer: Any) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        pad = max_len - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
        batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
        batch["labels"].append(item["labels"] + [-100] * pad)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_rows(parquet_dir: Path, manifest: Path, split: str, max_samples: int) -> list[dict[str, Any]]:
    splits = read_manifest_splits(manifest)
    frames = [pd.read_parquet(path, columns=["uid", "indication", "findings", "impression"]) for path in sorted(parquet_dir.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["uid"])
    if splits:
        uid_text = df["uid"].astype(str)
        case_ids = uid_text.where(uid_text.str.startswith("CXR"), "CXR" + uid_text)
        df = df[case_ids.map(splits).fillna("") == split]
    has_text = df["findings"].fillna("").astype(str).str.len().gt(0) | df["impression"].fillna("").astype(str).str.len().gt(0)
    df = df[has_text]
    df = df.sample(frac=1.0, random_state=20260714).head(max_samples)
    return df.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a small report generator LoRA on IU X-Ray reports.")
    parser.add_argument("--parquet-dir", default="/data/iu_xray_hf/data")
    parser.add_argument("--manifest", default="/outputs/manifest.jsonl")
    parser.add_argument("--output-dir", default="/outputs/lora/iu_xray_report_lora")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="development")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()

    rows = load_rows(Path(args.parquet_dir), Path(args.manifest), args.split, args.max_samples)
    if not rows:
        raise SystemExit("no training rows found")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    dataset = JsonReportDataset(rows, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda features: collate(features, tokenizer),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {"base_model": args.base_model, "split": args.split, "n_train": len(rows), "max_samples": args.max_samples}
    Path(args.output_dir, "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
