
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.pipeline import make_pipeline

from evaluation.generation import (
    _coerce_lora_report,
    _enforce_fewshot_checklist_report,
    _extract_json_object,
    _format_lora_prompt,
    _get_lora_generator,
)
from evaluation.models import EvaluationCase, record_from_state

LABEL_PATTERNS: list[tuple[str, list[str]]] = [
    ("normal", [r"\bnormal chest", r"no acute cardiopulmonary", r"no acute disease", r"no acute abnormality"]),
    ("cardiomegaly", [r"cardiomegaly", r"cardiac silhouette.*enlarg", r"heart size.*enlarg", r"mild enlargement of the cardiac"]),
    ("borderline cardiomegaly", [r"borderline cardiomegaly", r"borderline enlarged"]),
    ("hyperinflation", [r"hyperinflation", r"hyperinflated", r"emphysema", r"copd"]),
    ("atelectasis", [r"atelect", r"linear opacity", r"streaky opacity"]),
    ("opacity", [r"\bopacity", r"opacities", r"airspace disease", r"airspace opacity"]),
    ("infiltrate", [r"infiltrate", r"infiltration"]),
    ("consolidation", [r"consolidation", r"consolidative"]),
    ("pleural effusion", [r"pleural effusion", r"effusion"]),
    ("pneumothorax", [r"pneumothorax"]),
    ("pulmonary edema", [r"pulmonary edema", r"vascular congestion", r"interstitial edema"]),
    ("nodule", [r"\bnodule", r"nodular density", r"pulmonary nodule"]),
    ("granuloma", [r"granuloma", r"granulomatous"]),
    ("calcified granuloma", [r"calcified granuloma", r"calcified.*granuloma"]),
    ("scarring", [r"scarring", r"scar", r"fibrotic scar"]),
    ("fibrosis", [r"fibrosis", r"fibrotic", r"chronic interstitial"]),
    ("aortic atherosclerosis", [r"aortic atherosclerosis", r"atherosclerotic calcification", r"aortic calcification", r"calcified aorta"]),
    ("tortuous aorta", [r"tortuous aorta", r"aorta is tortuous", r"tortuosity"]),
    ("hiatal hernia", [r"hiatal hernia"]),
    ("sternotomy", [r"sternotomy", r"median sternotomy", r"postoperative changes"]),
    ("degenerative change", [r"degenerative", r"spondylosis", r"osteophyte"]),
    ("low lung volume", [r"low lung volume", r"low volume", r"poor inspiration"]),
    ("fracture", [r"fracture", r"compression deformity", r"wedging"]),
    ("mass", [r"\bmass", r"masslike"]),
]

NEGATIVE_LABELS = [
    "no focal consolidation",
    "no pleural effusion",
    "no pneumothorax",
    "no pulmonary edema",
]

POSITIVE_TO_SENTENCE = {
    "normal": "No acute cardiopulmonary abnormality is identified.",
    "cardiomegaly": "The cardiac silhouette is mildly enlarged.",
    "borderline cardiomegaly": "There is borderline enlargement of the cardiac silhouette.",
    "hyperinflation": "The lungs are mildly hyperinflated.",
    "atelectasis": "Mild linear atelectatic opacity is present.",
    "opacity": "Mild pulmonary opacity is present.",
    "infiltrate": "Mild pulmonary infiltrate is present.",
    "consolidation": "Focal airspace consolidation is present.",
    "pleural effusion": "A small pleural effusion is present.",
    "pneumothorax": "A pneumothorax is present.",
    "pulmonary edema": "Mild pulmonary edema is present.",
    "nodule": "A small pulmonary nodule is present.",
    "granuloma": "A granulomatous density is present.",
    "calcified granuloma": "A calcified granuloma is present.",
    "scarring": "Mild chronic scarring is present.",
    "fibrosis": "Mild chronic fibrotic change is present.",
    "aortic atherosclerosis": "Atherosclerotic calcification of the aortic arch is noted.",
    "tortuous aorta": "The thoracic aorta is tortuous.",
    "hiatal hernia": "A hiatal hernia is present.",
    "sternotomy": "Median sternotomy changes are noted.",
    "degenerative change": "Mild degenerative changes of the thoracic spine are present.",
    "low lung volume": "Low lung volumes are present.",
    "fracture": "A chronic osseous compression deformity is present.",
    "mass": "A masslike pulmonary opacity is present.",
}

LABEL_TO_ANATOMY = {
    "cardiomegaly": "heart", "borderline cardiomegaly": "heart",
    "aortic atherosclerosis": "mediastinum", "tortuous aorta": "mediastinum", "sternotomy": "mediastinum",
    "pleural effusion": "pleura", "pneumothorax": "pleura",
    "degenerative change": "bones", "fracture": "bones",
}


def case_uid(case_id: str) -> int | None:
    m = re.search(r"CXR(\d+)", str(case_id))
    return int(m.group(1)) if m else None


def clean_text(value: Any) -> str:
    text = str(value or "")
    return re.sub(r"\s+", " ", text.replace("XXXX", "")).strip()


def extract_labels(findings: Any, impression: Any) -> list[str]:
    text = clean_text(f"{findings} {impression}").lower()
    labels = []
    for label, patterns in LABEL_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            labels.append(label)
    if not labels:
        labels.append("normal")
    if len(labels) > 1 and "normal" in labels:
        labels.remove("normal")
    return labels


def image_features(blob: bytes | None) -> np.ndarray:
    if not blob:
        return np.zeros(32 * 32 + 12, dtype=np.float32)
    image = Image.open(io.BytesIO(blob)).convert("L").resize((32, 32))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=10, range=(0, 1), density=True)
    stats = np.array([arr.mean(), arr.std()], dtype=np.float32)
    return np.concatenate([arr.reshape(-1), hist.astype(np.float32), stats])


def row_features(row: dict[str, Any]) -> np.ndarray:
    frontal = image_features(row.get("img_frontal"))
    lateral = image_features(row.get("img_lateral"))
    return np.concatenate([frontal, lateral]).astype(np.float32)


def read_manifest_splits(path: Path) -> dict[str, str]:
    splits = {}
    if not path.exists():
        return splits
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        splits[str(row.get("case_id"))] = str(row.get("split", ""))
    return splits


def load_rows(parquet_dir: Path, manifest: Path, train_split: str, max_train: int) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    frames = [pd.read_parquet(p) for p in sorted(parquet_dir.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["uid"])
    df["_uid_int"] = df["uid"].astype(int)
    by_uid = {int(row["_uid_int"]): row for row in df.to_dict(orient="records")}
    splits = read_manifest_splits(manifest)
    if splits:
        case_ids = "CXR" + df["_uid_int"].astype(str)
        df["_split"] = case_ids.map(splits).fillna("")
        train_df = df[df["_split"] == train_split]
    else:
        train_df = df
    has_text = train_df["findings"].fillna("").astype(str).str.len().gt(0) | train_df["impression"].fillna("").astype(str).str.len().gt(0)
    train_df = train_df[has_text].sample(frac=1.0, random_state=20260715).head(max_train)
    return train_df.to_dict(orient="records"), by_uid


def train_vision_agent(rows: list[dict[str, Any]]):
    X = np.vstack([row_features(r) for r in rows])
    labels = [extract_labels(r.get("findings"), r.get("impression")) for r in rows]
    mlb = MultiLabelBinarizer(classes=[label for label, _ in LABEL_PATTERNS])
    Y = mlb.fit_transform(labels)
    # Drop columns with no positives in the chosen training subset; keep mapping for probabilities.
    active = np.where(Y.sum(axis=0) > 0)[0]
    active_classes = [mlb.classes_[i] for i in active]
    Ya = Y[:, active]
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X, Ya)
    return pipe, active_classes, X


def predict_labels(model, active_classes: list[str], x: np.ndarray, top_n: int) -> list[tuple[str, float]]:
    probs = model.predict_proba(x.reshape(1, -1))[0]
    ranked = sorted(zip(active_classes, probs), key=lambda p: p[1], reverse=True)
    positives = [(l, float(p)) for l, p in ranked if p >= 0.80 and l != "normal"][:top_n]
    if not positives:
        positives = [(l, float(p)) for l, p in ranked if p >= 0.55 and l != "normal"][: min(2, top_n)]
    normal_prob = next((float(p) for l, p in ranked if l == "normal"), 0.0)
    if normal_prob >= 0.55 and (not positives or positives[0][1] < 0.85):
        return [("normal", normal_prob)]
    return positives[:top_n] or [("normal", normal_prob)]


def original_agent_normal_prior(state: dict[str, Any]) -> bool:
    findings = state.get("image_findings", []) or []
    types = " ".join(str(item.get("finding_type", "")) for item in findings).lower()
    descriptions = " ".join(str(item.get("description", "")) for item in findings).lower()
    text = f"{types} {descriptions}"
    if "no_acute_abnormality" in text or "no acute abnormality" in text:
        return True
    return False


def reconcile_with_original_agent(predicted: list[tuple[str, float]], state: dict[str, Any]) -> list[tuple[str, float]]:
    if not original_agent_normal_prior(state):
        return predicted
    positives = [(label, score) for label, score in predicted if label != "normal"]
    if not positives:
        return predicted
    # The lightweight pixel classifier is intentionally conservative when it conflicts with
    # a strong no-acute-abnormality read from the upstream image agent.
    if max(score for _, score in positives) < 0.95:
        return [("normal", 0.80)]
    dangerous = {"pleural effusion", "pneumothorax", "consolidation", "pulmonary edema"}
    if any(label in dangerous for label, _ in positives) and len(positives) >= 2:
        return [("normal", 0.75)]
    return predicted


def retrieve_reports(train_rows: list[dict[str, Any]], train_X: np.ndarray, x: np.ndarray, k: int, exclude_uid: int | None) -> list[dict[str, Any]]:
    sims = cosine_similarity(x.reshape(1, -1), train_X)[0]
    order = np.argsort(-sims)
    out = []
    seen = set()
    for idx in order:
        row = train_rows[int(idx)]
        uid = int(row.get("_uid_int") or row.get("uid"))
        if exclude_uid is not None and uid == exclude_uid:
            continue
        if uid in seen:
            continue
        seen.add(uid)
        out.append({
            "uid": f"CXR{uid}",
            "similarity": round(float(sims[int(idx)]), 4),
            "labels": extract_labels(row.get("findings"), row.get("impression")),
            "findings": clean_text(row.get("findings")),
            "impression": clean_text(row.get("impression")),
        })
        if len(out) >= k:
            break
    return out


def structured_output(predicted: list[tuple[str, float]], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [label for label, _ in predicted if label != "normal"]
    if not positive:
        positive = []
    anatomy: dict[str, list[str]] = defaultdict(list)
    for label, score in predicted:
        if label == "normal":
            continue
        section = LABEL_TO_ANATOMY.get(label, "lungs")
        anatomy[section].append(POSITIVE_TO_SENTENCE.get(label, label))
    for neg in NEGATIVE_LABELS:
        if "consolidation" in neg and any(l in positive for l in ["consolidation", "infiltrate", "opacity"]):
            continue
        if "effusion" in neg and "pleural effusion" in positive:
            continue
        if "pneumothorax" in neg and "pneumothorax" in positive:
            continue
        if "edema" in neg and "pulmonary edema" in positive:
            continue
    negative = []
    if "consolidation" not in positive and "infiltrate" not in positive:
        negative.append("no focal consolidation")
    if "pleural effusion" not in positive:
        negative.append("no pleural effusion")
    if "pneumothorax" not in positive:
        negative.append("no pneumothorax")
    if "pulmonary edema" not in positive:
        negative.append("no pulmonary edema")
    if not positive:
        anatomy["lungs"].append("The lungs are clear without focal airspace disease.")
        anatomy["heart"].append("The cardiomediastinal silhouette is within normal size limits.")
    return {
        "agent_type": "fair_stage4_image_multilabel_agent",
        "source": "image_pixels_only_at_test_time; labels learned from training reports",
        "positive_labels": positive,
        "negative_labels": negative,
        "confidence": {label: round(float(score), 4) for label, score in predicted},
        "anatomy_findings": dict(anatomy),
        "impression_keywords": positive[:4] or ["no acute cardiopulmonary abnormality"],
        "retrieval_summary": [{"uid": r["uid"], "similarity": r["similarity"], "labels": r["labels"]} for r in retrieved],
    }


def findings_for_state(vision: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    i = 1
    for section, sentences in vision.get("anatomy_findings", {}).items():
        for sentence in sentences:
            out.append({
                "finding_id": f"STAGE4-VISION-{i}",
                "finding_type": section,
                "location": section,
                "description": sentence,
                "confidence": max(vision.get("confidence", {}).values() or [0.5]),
            })
            i += 1
    for label in vision.get("negative_labels", []):
        out.append({
            "finding_id": f"STAGE4-VISION-{i}",
            "finding_type": label.replace(" ", "_"),
            "location": "chest",
            "description": label,
            "confidence": 0.8,
        })
        i += 1
    return out


def evidence_for_state(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(retrieved, start=1):
        out.append({
            "evidence_id": f"STAGE4-RETRIEVAL-{i}",
            "title": f"Image-similar training report {row['uid']}",
            "summary": f"FINDINGS: {row['findings']} IMPRESSION: {row['impression']}",
            "source": "development_split_image_retrieval",
            "similarity": row["similarity"],
        })
    return out


def lora_payload_stage4(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_uid": state["study_uid"],
        "modality": state.get("modality", "DX"),
        "body_part": state.get("body_part", "CHEST"),
        "clinical_context": state.get("clinical_context", {}),
        "vision_agent_structured_output": state.get("vision_agent_structured_output", {}),
        "image_findings": state.get("image_findings", []),
        "retrieved_similar_reports": state.get("retrieved_similar_reports", []),
        "retrieved_evidence": state.get("retrieved_evidence", []),
        "required_output_style": {
            "language": "English",
            "sections": ["FINDINGS", "IMPRESSION"],
            "instruction": (
                "Use the vision_agent_structured_output as primary evidence. Use retrieved_similar_reports only as IU style and wording references. "
                "Prefer explicit mention of predicted positive labels when confidence is moderate/high. Do not add unsupported findings. "
                "Return valid ReportDraft JSON."
            ),
        },
    }


def deterministic_stage4_report(state: dict[str, Any]):
    from app.schemas import ReportDraft
    from evaluation.generation import _statement, _first_evidence_id

    evidence_id = _first_evidence_id(state)
    vision = state.get("vision_agent_structured_output", {})
    positives = [p for p in vision.get("positive_labels", []) if p != "normal"]
    anatomy = vision.get("anatomy_findings", {}) or {}
    negative = set(vision.get("negative_labels", []) or [])
    findings: list[Any] = []

    heart_items = anatomy.get("heart", [])
    mediastinum_items = anatomy.get("mediastinum", [])
    lung_items = anatomy.get("lungs", [])
    pleura_items = anatomy.get("pleura", [])
    bone_items = anatomy.get("bones", [])

    if heart_items:
        findings.extend(_statement(text, evidence_id) for text in heart_items)
    else:
        findings.append(_statement("The cardiomediastinal silhouette and mediastinum are within normal size limits.", evidence_id))
    findings.extend(_statement(text, evidence_id) for text in mediastinum_items)

    if lung_items:
        findings.extend(_statement(text, evidence_id) for text in lung_items)
    else:
        findings.append(_statement("The lungs are clear.", evidence_id))

    if "no pulmonary edema" in negative:
        findings.append(_statement("There is no pulmonary edema.", evidence_id))
    if "no focal consolidation" in negative:
        findings.append(_statement("There is no focal consolidation.", evidence_id))
    findings.extend(_statement(text, evidence_id) for text in pleura_items)
    if "no pleural effusion" in negative:
        findings.append(_statement("There is no pleural effusion.", evidence_id))
    if "no pneumothorax" in negative:
        findings.append(_statement("There is no pneumothorax.", evidence_id))
    findings.extend(_statement(text, evidence_id) for text in bone_items)

    if positives:
        phrase = ", ".join(positives[:3])
        impression = [_statement(f"{phrase.capitalize()}.", evidence_id)]
    else:
        impression = [_statement("No acute cardiopulmonary abnormality.", evidence_id)]

    return ReportDraft(
        clinical_history=str(state.get("clinical_context", {}).get("chief_complaint") or "Not provided"),
        technique="Chest radiographs.",
        findings=findings,
        impression=impression,
        recommendations=[],
        uncertainties=[],
        ai_disclaimer="AI-generated draft report for physician review only.",
    )


def report_text_is_bad(state: dict[str, Any]) -> bool:
    draft = state.get("report_draft", {}) or {}
    text = " ".join(str(item.get("text", "")) for item in draft.get("findings", []) + draft.get("impression", []))
    lowered = text.lower()
    if len(text) > 1800:
        return True
    if text.count("No acute radiographic findings") >= 3:
        return True
    if "{\"clinical_history\"" in text or "\"findings\"" in text:
        return True
    if "sputum" in lowered or "blood" in lowered:
        return True
    return False


def consistency_cleanup(state: dict[str, Any]) -> None:
    from app.schemas import ReportDraft
    from evaluation.generation import _statement, _first_evidence_id
    draft = ReportDraft.model_validate(state["report_draft"])
    evidence_id = _first_evidence_id(state)
    text = " ".join([x.text for x in draft.findings + draft.impression]).lower()
    vision = state.get("vision_agent_structured_output", {})
    positives = [p for p in vision.get("positive_labels", []) if p != "normal"]
    for label in positives[:4]:
        key = label.split()[0]
        if key not in text:
            draft.findings.append(_statement(POSITIVE_TO_SENTENCE.get(label, label), evidence_id))
            text += " " + label
    negs = vision.get("negative_labels", [])
    if "no pleural effusion" in negs and "pleural effusion" not in positives and "pleural effusion" not in text:
        draft.findings.append(_statement("There is no pleural effusion.", evidence_id))
    if "no pneumothorax" in negs and "pneumothorax" not in positives and "pneumothorax" not in text:
        draft.findings.append(_statement("There is no pneumothorax.", evidence_id))
    if positives and (not draft.impression or "no acute cardiopulmonary abnormality" in " ".join(i.text.lower() for i in draft.impression)):
        phrase = ", ".join(positives[:3])
        draft.impression = [_statement(f"{phrase.capitalize()}. No acute cardiopulmonary abnormality otherwise identified.", evidence_id)]
    # Remove exact repeated sentences before applying the existing IU checklist guard.
    seen = set()
    cleaned = []
    for item in draft.findings:
        key = re.sub(r"\s+", " ", item.text.strip().lower())
        if key and key not in seen:
            cleaned.append(item)
            seen.add(key)
    draft.findings = cleaned[:12]
    state["report_draft"] = draft.model_dump()
    _enforce_fewshot_checklist_report(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair Stage 4: image multilabel agent + image retrieval + IU template + LoRA writer.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--source-method", default="full_english_template")
    parser.add_argument("--output", required=True)
    parser.add_argument("--parquet-dir", default="/data/iu_xray_hf/data")
    parser.add_argument("--manifest", default="/outputs/manifest.jsonl")
    parser.add_argument("--train-split", default="development")
    parser.add_argument("--max-train", type=int, default=512)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-labels", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    train_rows, by_uid = load_rows(Path(args.parquet_dir), Path(args.manifest), args.train_split, args.max_train)
    if not train_rows:
        raise SystemExit("No training rows for stage4 vision agent")
    vision_model, active_classes, train_X = train_vision_agent(train_rows)
    tokenizer, model = _get_lora_generator()

    rows = []
    for line in Path(args.records).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("method_id") == args.source_method and row.get("success"):
                rows.append(row)
    rows = rows[:args.limit]

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            started = time.perf_counter()
            uid = case_uid(row.get("case_id", ""))
            data_row = by_uid.get(uid or -1)
            if data_row is None:
                raise RuntimeError(f"No parquet image row for {row.get('case_id')}")
            state = dict(row["state"])
            x = row_features(data_row)
            predicted = predict_labels(vision_model, active_classes, x, args.top_labels)
            predicted = reconcile_with_original_agent(predicted, state)
            retrieved = retrieve_reports(train_rows, train_X, x, args.top_k, uid)
            vision = structured_output(predicted, retrieved)

            state["vision_agent_structured_output"] = vision
            state["retrieved_similar_reports"] = retrieved
            state["image_findings"] = findings_for_state(vision)
            state["retrieved_evidence"] = evidence_for_state(retrieved)
            payload = lora_payload_stage4(state)
            state["text_agent_input"] = payload
            prompt = _format_lora_prompt(payload)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            state["text_agent_raw_output"] = generated
            try:
                raw_report = json.loads(_extract_json_object(generated))
            except Exception:
                raw_report = None
            if raw_report is None:
                draft = deterministic_stage4_report(state)
                state["text_agent_fallback_reason"] = "lora_output_not_valid_json"
            else:
                draft = _coerce_lora_report(raw_report, state)
            state["report_draft"] = draft.model_dump()
            if report_text_is_bad(state):
                state["report_draft"] = deterministic_stage4_report(state).model_dump()
                state["text_agent_fallback_reason"] = "lora_output_failed_cleanliness_checks"
            consistency_cleanup(state)
            case = EvaluationCase(
                case_id=row["case_id"],
                split=row.get("split", "test"),
                stratum=row.get("stratum", "unspecified"),
                image_paths=row.get("image_paths", []),
                reference_findings=row.get("reference_findings", ""),
                reference_impression=row.get("reference_impression", ""),
            )
            record = record_from_state(case, "stage4_vision_rag_lora", state, time.perf_counter() - started)
            record.metadata.update({
                "fairness": "test-time input uses image pixels only; no test MeSH/Problems/reference used for generation",
                "vision_agent": "logistic multilabel classifier on image features, supervised by development report-derived labels",
                "retrieval": "cosine retrieval over development image features",
                "lora_adapter": os.environ.get("LORA_REPORT_MODEL_PATH", "/outputs/lora/iu_xray_report_lora_clean_retrieval_512"),
                "max_train": args.max_train,
            })
            output.write(record.model_dump_json() + "\n")
            print(f"{record.case_id} stage4_vision_rag_lora: ok labels={vision['positive_labels']}", flush=True)


if __name__ == "__main__":
    main()
