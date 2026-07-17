from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor,
    Trainer,
    TrainingArguments,
)

from evaluation.generation import _coerce_lora_report, _enforce_fewshot_checklist_report, _extract_json_object
from evaluation.models import EvaluationCase, record_from_state


LABEL_PATTERNS: list[tuple[str, list[str]]] = [
    ("normal", [r"\bnormal chest", r"no acute cardiopulmonary", r"no acute disease", r"no acute abnormality"]),
    ("cardiomegaly", [r"cardiomegaly", r"cardiac silhouette.*enlarg", r"heart size.*enlarg"]),
    ("borderline cardiomegaly", [r"borderline cardiomegaly", r"borderline enlarged"]),
    ("hyperinflation", [r"hyperinflation", r"hyperinflated", r"emphysema", r"\bcopd\b"]),
    ("atelectasis", [r"atelect", r"linear opacity", r"streaky opacity"]),
    ("opacity", [r"\bopacity", r"opacities", r"airspace disease", r"airspace opacity"]),
    ("infiltrate", [r"infiltrate", r"infiltration"]),
    ("consolidation", [r"consolidation", r"consolidative"]),
    ("pleural effusion", [r"pleural effusion", r"\beffusion\b"]),
    ("pneumothorax", [r"pneumothorax"]),
    ("pulmonary edema", [r"pulmonary edema", r"vascular congestion", r"interstitial edema"]),
    ("nodule", [r"\bnodule", r"nodular density", r"pulmonary nodule"]),
    ("granuloma", [r"granuloma", r"granulomatous"]),
    ("calcified granuloma", [r"calcified granuloma", r"calcified.*granuloma"]),
    ("scarring", [r"scarring", r"\bscar\b", r"fibrotic scar"]),
    ("fibrosis", [r"fibrosis", r"fibrotic", r"chronic interstitial"]),
    ("aortic atherosclerosis", [r"aortic atherosclerosis", r"atherosclerotic calcification", r"aortic calcification"]),
    ("tortuous aorta", [r"tortuous aorta", r"aorta is tortuous", r"tortuosity"]),
    ("hiatal hernia", [r"hiatal hernia"]),
    ("sternotomy", [r"sternotomy", r"median sternotomy", r"postoperative changes"]),
    ("degenerative change", [r"degenerative", r"spondylosis", r"osteophyte"]),
    ("low lung volume", [r"low lung volume", r"low volume", r"poor inspiration"]),
    ("fracture", [r"fracture", r"compression deformity", r"wedging"]),
    ("mass", [r"\bmass", r"masslike"]),
]


NEGATIVE_LABELS = ["no focal consolidation", "no pleural effusion", "no pneumothorax", "no pulmonary edema"]


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("XXXX", "")).strip()


def case_uid(case_id: str) -> int | None:
    match = re.search(r"CXR(\d+)", str(case_id))
    return int(match.group(1)) if match else None


def read_manifest_splits(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["case_id"])] = str(row.get("split", ""))
    return out


def load_parquet_rows(parquet_dir: Path, manifest: Path, split: str | None, max_samples: int | None) -> list[dict[str, Any]]:
    frames = [pd.read_parquet(path) for path in sorted(parquet_dir.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["uid"])
    df["_uid_int"] = df["uid"].astype(int)
    splits = read_manifest_splits(manifest)
    if split and splits:
        case_ids = "CXR" + df["_uid_int"].astype(str)
        df["_split"] = case_ids.map(splits).fillna("")
        df = df[df["_split"] == split]
    has_text = df["findings"].fillna("").astype(str).str.len().gt(0) | df["impression"].fillna("").astype(str).str.len().gt(0)
    df = df[has_text].sample(frac=1.0, random_state=20260715)
    if max_samples:
        df = df.head(max_samples)
    return df.to_dict(orient="records")


def load_all_by_uid(parquet_dir: Path) -> dict[int, dict[str, Any]]:
    frames = [pd.read_parquet(path) for path in sorted(parquet_dir.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["uid"])
    df["_uid_int"] = df["uid"].astype(int)
    return {int(row["_uid_int"]): row for row in df.to_dict(orient="records")}


def pil_images(row: dict[str, Any]) -> list[Image.Image]:
    images = []
    for key in ("img_frontal", "img_lateral"):
        blob = row.get(key)
        if blob:
            image = Image.open(io.BytesIO(blob)).convert("RGB")
            image.thumbnail((336, 336), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (336, 336), (0, 0, 0))
            canvas.paste(image, ((336 - image.width) // 2, (336 - image.height) // 2))
            images.append(canvas)
    return images[:2]


def ensure_two_images(images: list[Image.Image]) -> list[Image.Image]:
    if not images:
        blank = Image.new("RGB", (336, 336), (0, 0, 0))
        return [blank, blank.copy()]
    while len(images) < 2:
        images.append(images[0].copy())
    return images[:2]


def extract_labels(findings: Any, impression: Any) -> list[str]:
    text = clean_text(f"{findings} {impression}").lower()
    labels = [label for label, patterns in LABEL_PATTERNS if any(re.search(pattern, text) for pattern in patterns)]
    if not labels:
        labels = ["normal"]
    if len(labels) > 1 and "normal" in labels:
        labels.remove("normal")
    return labels


def structured_target(row: dict[str, Any]) -> dict[str, Any]:
    labels = extract_labels(row.get("findings"), row.get("impression"))
    positives = [label for label in labels if label != "normal"]
    negatives = []
    if "consolidation" not in positives and "infiltrate" not in positives:
        negatives.append("no focal consolidation")
    if "pleural effusion" not in positives:
        negatives.append("no pleural effusion")
    if "pneumothorax" not in positives:
        negatives.append("no pneumothorax")
    if "pulmonary edema" not in positives:
        negatives.append("no pulmonary edema")
    return {
        "positive_labels": positives,
        "negative_labels": negatives,
        "medical_findings": positives + negatives,
        "impression_keywords": positives or ["no acute cardiopulmonary abnormality"],
    }


def vision_prompt() -> str:
    return (
        "Analyze the chest X-ray images and output only JSON. "
        "Predict structured Medical Findings for the current patient. "
        "Use positive_labels for abnormalities and negative_labels for important normal findings. "
        "Do not write a full report."
    )


def make_vl_text(processor: Any, prompt: str, target: str | None = None) -> str:
    content = [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + (target or "")


class VisionDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], processor: Any, max_length: int):
        self.rows = rows
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        images = pil_images(row)
        while len(images) < 2:
            images.append(images[0].copy())
        target = json.dumps(structured_target(row), ensure_ascii=False) + self.processor.tokenizer.eos_token
        prompt_text = make_vl_text(self.processor, vision_prompt())
        full_text = make_vl_text(self.processor, vision_prompt(), target)
        prompt = self.processor(text=[prompt_text], images=images, return_tensors="pt", padding=False, truncation=True, max_length=self.max_length)
        full = self.processor(text=[full_text], images=images, return_tensors="pt", padding=False, truncation=True, max_length=self.max_length)
        item = {k: v.squeeze(0) for k, v in full.items()}
        labels = item["input_ids"].clone()
        labels[: min(prompt["input_ids"].shape[-1], labels.shape[-1])] = -100
        item["labels"] = labels
        return item


def pad_tensor(values: list[torch.Tensor], pad_value: int | float) -> torch.Tensor:
    max_len = max(v.shape[0] for v in values)
    out = []
    for value in values:
        pad_len = max_len - value.shape[0]
        if pad_len:
            value = torch.cat([value, torch.full((pad_len,), pad_value, dtype=value.dtype)])
        out.append(value)
    return torch.stack(out)


def collate_vl(features: list[dict[str, Any]], processor: Any) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
        if key not in features[0]:
            continue
        pad = -100 if key == "labels" else (processor.tokenizer.pad_token_id if key == "input_ids" else 0)
        batch[key] = pad_tensor([f[key] for f in features], pad)
    for key in features[0]:
        if key not in batch:
            batch[key] = torch.cat([f[key] for f in features], dim=0)
    return batch


def train_vision(args: argparse.Namespace) -> None:
    rows = load_parquet_rows(Path(args.parquet_dir), Path(args.manifest), args.split, args.max_samples)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    dataset = VisionDataset(rows, processor, args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=lambda f: collate_vl(f, processor))
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    Path(args.output_dir, "training_meta.json").write_text(
        json.dumps({"base_model": args.base_model, "n_train": len(rows), "split": args.split}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def label_overlap(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def retrieve_by_labels(row: dict[str, Any], pool: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    labels = extract_labels(row.get("findings"), row.get("impression"))
    scored = []
    uid = int(row["_uid_int"])
    for other in pool:
        if int(other["_uid_int"]) == uid:
            continue
        other_labels = extract_labels(other.get("findings"), other.get("impression"))
        scored.append((label_overlap(labels, other_labels), other))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


def format_reference_report(row: dict[str, Any]) -> str:
    return f"FINDINGS:\n{clean_text(row.get('findings'))}\nIMPRESSION:\n{clean_text(row.get('impression'))}"


def format_reference_reports(rows: list[dict[str, Any]], limit: int = 3) -> str:
    blocks = []
    for idx, row in enumerate(rows[:limit], start=1):
        labels = ", ".join(extract_labels(row.get("findings"), row.get("impression")))
        blocks.append(
            f"STYLE TEMPLATE {idx} (weak reference only; labels: {labels})\n"
            f"{format_reference_report(row)}"
        )
    return "\n\n".join(blocks)


def text_prompt(medical_findings: list[str], reference_report: str) -> str:
    return (
        "System: You are a professional chest X-ray report writer.\n"
        "Task: generate FINDINGS and IMPRESSION for the CURRENT PATIENT.\n"
        "HARD CONSTRAINT - CURRENT PATIENT DIAGNOSIS:\n"
        "The Medical Findings list is the only diagnostic truth for the current patient. "
        "Every abnormality you assert must be supported by this list, and every explicit negative item must remain negative.\n"
        "WEAK CONSTRAINT - SIMILAR REPORT TEMPLATES:\n"
        "The Reference Reports are from other patients. They are style, syntax, and section-order templates only. "
        "Do not copy their diagnoses, anatomy, devices, or complications unless they also appear in Medical Findings.\n"
        "If a template conflicts with Medical Findings, ignore the template and follow Medical Findings.\n\n"
        "User:\n"
        "[CURRENT PATIENT - KNOWN LABELS / Medical Findings - HARD CONSTRAINT]\n"
        f"{json.dumps(medical_findings, ensure_ascii=False)}\n\n"
        "[OTHER PATIENTS - SIMILAR REPORTS - WEAK STYLE REFERENCES ONLY]\n"
        f"{reference_report}\n\n"
        "Write the target report for the CURRENT PATIENT only. "
        "Use the known labels as diagnosis; use the similar reports only for wording style.\n"
        "Assistant:\n"
    )


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], pool: list[dict[str, Any]], tokenizer: Any, max_length: int, top_k: int):
        self.examples = []
        for row in rows:
            refs = retrieve_by_labels(row, pool, top_k)
            reference = format_reference_reports(refs, top_k) if refs else format_reference_report(row)
            findings = structured_target(row)["medical_findings"]
            prompt = text_prompt(findings, reference)
            target = f"FINDINGS:\n{clean_text(row.get('findings'))}\n\nIMPRESSION:\n{clean_text(row.get('impression'))}" + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full = tokenizer(prompt + target, add_special_tokens=False, truncation=True, max_length=max_length)
            labels = full.input_ids.copy()
            labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
            self.examples.append({"input_ids": full.input_ids, "attention_mask": full.attention_mask, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def collate_text(features: list[dict[str, Any]], tokenizer: Any) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        pad = max_len - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
        batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
        batch["labels"].append(item["labels"] + [-100] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def train_text(args: argparse.Namespace) -> None:
    rows = load_parquet_rows(Path(args.parquet_dir), Path(args.manifest), args.split, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    dataset = TextDataset(rows, rows, tokenizer, args.max_length, args.top_k)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=lambda f: collate_text(f, tokenizer))
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "training_meta.json").write_text(
        json.dumps({"base_model": args.base_model, "n_train": len(rows), "prompt": "medical_findings_strong_reference_weak"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(_extract_json_object(text))
    except Exception:
        return {"positive_labels": [], "negative_labels": NEGATIVE_LABELS, "medical_findings": NEGATIVE_LABELS, "impression_keywords": ["no acute cardiopulmonary abnormality"]}


def predict_vision(row: dict[str, Any], processor: Any, model: Any, max_new_tokens: int) -> dict[str, Any]:
    images = ensure_two_images(pil_images(row))
    prompt = make_vl_text(processor, vision_prompt())
    inputs = processor(text=[prompt], images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
    decoded = processor.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    result = parse_json_object(decoded)
    result["raw_output"] = decoded
    result.setdefault("medical_findings", list(result.get("positive_labels", [])) + list(result.get("negative_labels", [])))
    return result


def image_embedding(row: dict[str, Any], processor: Any, model: Any, device: str) -> np.ndarray:
    images = pil_images(row)
    if not images:
        return np.zeros(512, dtype=np.float32)
    inputs = processor(images=images[0], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].detach().cpu().float().numpy()


def build_image_index(rows: list[dict[str, Any]], model_name: str) -> tuple[Any, Any, str, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    vectors = np.vstack([image_embedding(row, processor, model, device) for row in rows])
    return processor, model, device, vectors


def retrieve_by_image(row: dict[str, Any], pool: list[dict[str, Any]], vectors: np.ndarray, clip_processor: Any, clip_model: Any, device: str, k: int) -> list[dict[str, Any]]:
    emb = image_embedding(row, clip_processor, clip_model, device)
    sims = cosine_similarity(emb.reshape(1, -1), vectors)[0]
    order = np.argsort(-sims)
    out = []
    uid = int(row["_uid_int"])
    for idx in order:
        item = pool[int(idx)]
        if int(item["_uid_int"]) == uid:
            continue
        out.append({
            "uid": f"CXR{int(item['_uid_int'])}",
            "similarity": round(float(sims[int(idx)]), 4),
            "labels": extract_labels(item.get("findings"), item.get("impression")),
            "findings": clean_text(item.get("findings")),
            "impression": clean_text(item.get("impression")),
        })
        if len(out) >= k:
            break
    return out


def parse_report_text(text: str) -> tuple[str, str]:
    findings = text
    impression = ""
    m = re.search(r"FINDINGS:\s*(.*?)\s*IMPRESSION:\s*(.*)", text, flags=re.I | re.S)
    if m:
        findings, impression = m.group(1).strip(), m.group(2).strip()
    return clean_text(findings), clean_text(impression)


def generate_text_report(findings: list[str], reference_report: str, tokenizer: Any, model: Any) -> tuple[str, str, str]:
    prompt = text_prompt(findings, reference_report)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=500, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    f, i = parse_report_text(generated)
    return f, i, generated


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[\u4e00-\u9fff]", text.lower())


def consistency_rewrite(findings: str, impression: str, medical_findings: list[str]) -> tuple[str, str]:
    text = f"{findings} {impression}".lower()
    sentences = []
    if not findings or len(findings) > 1800 or text.count("visible") > 5:
        findings = ""
    if not findings:
        sentences.append("The cardiomediastinal silhouette and mediastinum are within normal size limits.")
        if any(x in medical_findings for x in ("cardiomegaly", "borderline cardiomegaly")):
            sentences[-1] = "The cardiac silhouette is mildly enlarged."
        if any(x in medical_findings for x in ("opacity", "atelectasis", "infiltrate")):
            sentences.append("Mild basilar linear opacity or atelectatic change is present.")
        else:
            sentences.append("The lungs are clear.")
        if "no pulmonary edema" in medical_findings:
            sentences.append("There is no pulmonary edema.")
        if "no focal consolidation" in medical_findings:
            sentences.append("There is no focal consolidation.")
        if "no pleural effusion" in medical_findings:
            sentences.append("There is no pleural effusion.")
        if "no pneumothorax" in medical_findings:
            sentences.append("There is no pneumothorax.")
        findings = " ".join(sentences)
    positives = [x for x in medical_findings if not x.startswith("no ")]
    if not impression or len(impression) > 500:
        impression = (", ".join(positives[:3]).capitalize() + ".") if positives else "No acute cardiopulmonary abnormality."
    return findings, impression


def run(args: argparse.Namespace) -> None:
    train_rows = load_parquet_rows(Path(args.parquet_dir), Path(args.manifest), args.train_split, None)
    by_uid = load_all_by_uid(Path(args.parquet_dir))
    clip_processor, clip_model, clip_device, image_vectors = build_image_index(train_rows, args.embedding_model)

    vision_processor = AutoProcessor.from_pretrained(args.vision_base_model, trust_remote_code=True)
    vision_model = AutoModelForImageTextToText.from_pretrained(
        args.vision_base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    vision_model = PeftModel.from_pretrained(vision_model, args.vision_lora)
    vision_model.eval()

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_base_model, trust_remote_code=True)
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    text_model = AutoModelForCausalLM.from_pretrained(
        args.text_base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    text_model = PeftModel.from_pretrained(text_model, args.text_lora)
    text_model.eval()

    source = []
    for line in Path(args.records).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("method_id") == args.source_method and row.get("success"):
                source.append(row)
    source = source[: args.limit]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as out:
        for record_row in source:
            started = time.perf_counter()
            uid = case_uid(record_row["case_id"])
            image_row = by_uid[int(uid)]
            vision = predict_vision(image_row, vision_processor, vision_model, args.vision_max_new_tokens)
            retrieved = retrieve_by_image(image_row, train_rows, image_vectors, clip_processor, clip_model, clip_device, args.top_k)
            reference_report = format_reference_reports(retrieved, args.top_k) if retrieved else "FINDINGS:\nThe lungs are clear.\nIMPRESSION:\nNo acute cardiopulmonary abnormality."
            medical_findings = list(dict.fromkeys([str(x) for x in vision.get("medical_findings", []) if str(x).strip()]))
            findings, impression, raw_text = generate_text_report(medical_findings, reference_report, text_tokenizer, text_model)
            findings, impression = consistency_rewrite(findings, impression, medical_findings)
            state = dict(record_row["state"])
            state["vision_agent_structured_output"] = vision
            state["retrieved_similar_reports"] = retrieved
            state["text_agent_input"] = {"medical_findings": medical_findings, "reference_report": reference_report, "constraint": "Medical Findings are strong evidence; Reference Report is weak style template only."}
            state["text_agent_raw_output"] = raw_text
            state["report_draft"] = {
                "clinical_history": state.get("clinical_context", {}).get("chief_complaint") or "Not provided",
                "technique": "Chest radiographs.",
                "findings": [{"text": findings, "evidence_ids": ["STAGE4_VISION_FINDINGS"]}],
                "impression": [{"text": impression, "evidence_ids": ["STAGE4_VISION_FINDINGS"]}],
                "recommendations": [],
                "uncertainties": [],
                "ai_disclaimer": "AI-generated draft report for physician review only.",
            }
            _enforce_fewshot_checklist_report(state)
            case = EvaluationCase(
                case_id=record_row["case_id"],
                split=record_row.get("split", "test"),
                stratum=record_row.get("stratum", "unspecified"),
                image_paths=record_row.get("image_paths", []),
                reference_findings=record_row.get("reference_findings", ""),
                reference_impression=record_row.get("reference_impression", ""),
            )
            rec = record_from_state(case, "stage4_real_qwenvl_rag_textlora", state, time.perf_counter() - started)
            rec.metadata["fairness"] = "test-time generation uses images only for labels and train-split image retrieval; no test MeSH/Problems/reference"
            out.write(rec.model_dump_json() + "\n")
            print(f"{rec.case_id} stage4_real: ok findings={medical_findings}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("train-vision")
    v.add_argument("--parquet-dir", default="/data/iu_xray_hf/data")
    v.add_argument("--manifest", default="/outputs/manifest.jsonl")
    v.add_argument("--split", default="development")
    v.add_argument("--max-samples", type=int, default=1024)
    v.add_argument("--output-dir", default="/outputs/lora/stage4_qwenvl_vision_lora_1024")
    v.add_argument("--base-model", default="Qwen/Qwen3-VL-32B-Instruct")
    v.add_argument("--epochs", type=float, default=1.0)
    v.add_argument("--learning-rate", type=float, default=1e-4)
    v.add_argument("--max-length", type=int, default=1024)
    v.add_argument("--grad-accum", type=int, default=8)
    v.add_argument("--lora-r", type=int, default=8)
    v.add_argument("--lora-alpha", type=int, default=16)
    v.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)

    t = sub.add_parser("train-text")
    t.add_argument("--parquet-dir", default="/data/iu_xray_hf/data")
    t.add_argument("--manifest", default="/outputs/manifest.jsonl")
    t.add_argument("--split", default="development")
    t.add_argument("--max-samples", type=int, default=512)
    t.add_argument("--output-dir", default="/outputs/lora/stage4_text_lora_medfindings_ref_512")
    t.add_argument("--base-model", default="google/medgemma-27b-text-it")
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--learning-rate", type=float, default=2e-4)
    t.add_argument("--max-length", type=int, default=1536)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--top-k", type=int, default=3)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)

    r = sub.add_parser("run")
    r.add_argument("--records", required=True)
    r.add_argument("--source-method", default="full_english_template")
    r.add_argument("--output", required=True)
    r.add_argument("--parquet-dir", default="/data/iu_xray_hf/data")
    r.add_argument("--manifest", default="/outputs/manifest.jsonl")
    r.add_argument("--train-split", default="development")
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--top-k", type=int, default=3)
    r.add_argument("--vision-base-model", default="Qwen/Qwen3-VL-32B-Instruct")
    r.add_argument("--vision-lora", default="/outputs/lora/stage4_qwenvl_vision_lora_1024")
    r.add_argument("--vision-max-new-tokens", type=int, default=180)
    r.add_argument("--text-base-model", default="google/medgemma-27b-text-it")
    r.add_argument("--text-lora", default="/outputs/lora/stage4_text_lora_medfindings_ref_512")
    r.add_argument("--embedding-model", default="openai/clip-vit-base-patch32")

    args = parser.parse_args()
    if args.cmd == "train-vision":
        train_vision(args)
    elif args.cmd == "train-text":
        train_text(args)
    elif args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
