
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from evaluation.generation import _coerce_lora_report, _enforce_fewshot_checklist_report, _extract_json_object, _get_lora_generator
from evaluation.models import EvaluationCase, record_from_state

_WORD_RE = re.compile(r"[a-z0-9]+")
_ARTIFACT_RE = re.compile(r"\bF-[0-9a-f]{6,}\b|\[\]|\bNone\b|\b1\.0\b|```(?:json)?|```")
STOP = {"the", "and", "with", "without", "there", "are", "is", "of", "a", "an", "to", "in", "for", "no"}


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = _ARTIFACT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in STOP}


def state_text(state: dict[str, Any]) -> str:
    parts = [json.dumps(state.get("clinical_context", {}), ensure_ascii=False)]
    for finding in state.get("image_findings", []) or []:
        if isinstance(finding, dict):
            parts.extend(str(finding.get(k, "")) for k in ("finding_type", "location", "description", "text"))
        else:
            parts.append(str(finding))
    return clean_text(" ".join(parts))


def load_index(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["token_set"] = set(row.get("tokens") or tokenize(" ".join(str(row.get(k, "")) for k in ("indication", "mesh", "problems", "findings", "impression"))))
    return rows


def retrieve(index: list[dict[str, Any]], state: dict[str, Any], limit: int = 4) -> list[dict[str, str]]:
    query_text = state_text(state)
    query = tokenize(query_text)
    scored = []
    for row in index:
        overlap = len(query & row.get("token_set", set()))
        bonus = 0
        lower = query_text.lower()
        report = (row.get("findings", "") + " " + row.get("impression", "")).lower()
        for term in ("cardiomegaly", "effusion", "pneumothorax", "opacity", "atelectasis", "edema", "consolidation", "fracture"):
            if term in lower and term in report:
                bonus += 3
        score = overlap + bonus
        if score:
            scored.append((score, row.get("case_id", ""), row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    examples = [{"findings": clean_text(item[2].get("findings")), "impression": clean_text(item[2].get("impression"))} for item in scored[:limit]]
    defaults = [
        {"findings": "The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There is no pleural effusion. There is no evidence of pneumothorax.", "impression": "Normal chest x-XXXX."},
        {"findings": "Heart size and mediastinal contours are normal. The lungs are clear. No focal airspace consolidation, pleural effusion, or pneumothorax.", "impression": "No acute cardiopulmonary abnormality."},
    ]
    while len(examples) < limit:
        examples.append(defaults[len(examples) % len(defaults)])
    return examples


def format_prompt(state: dict[str, Any], examples: list[dict[str, str]]) -> str:
    payload = {
        "study_uid": state.get("study_uid"),
        "clinical_context": state.get("clinical_context", {}),
        "structured_image_findings": state.get("image_findings", []),
        "retrieved_iu_xray_style_examples": examples,
        "required_output_style": {
            "language": "English",
            "sections": ["FINDINGS", "IMPRESSION"],
            "style": "IU X-Ray reference-report wording; short declarative sentences; include common negative chest findings when appropriate; do not emit internal ids, JSON state, confidences, None, or arrays.",
        },
    }
    return "You are an attending radiologist. Write an IU X-Ray style chest radiograph report.\nUse the structured image findings and retrieved IU X-Ray examples as guidance.\nReturn only valid compact JSON with exactly two string keys: findings and impression.\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False) + "\n\nOUTPUT_JSON:\n"


def parse_report(generated: str) -> dict[str, Any]:
    try:
        raw = json.loads(_extract_json_object(generated))
    except Exception:
        raw = {"findings": clean_text(generated), "impression": ""}
    findings = clean_text(raw.get("findings") or raw.get("FINDINGS") or "")
    impression = clean_text(raw.get("impression") or raw.get("IMPRESSION") or "")
    if len(findings) > 1200:
        findings = findings[:1200].rsplit(" ", 1)[0]
    if len(impression) > 500:
        impression = impression[:500].rsplit(" ", 1)[0]
    return {"findings": findings, "impression": impression}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrieval-augmented clean LoRA third-stage reports from saved full-agent states.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--source-method", default="full_english_template")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--retrieval-index", default=os.environ.get("LORA_RETRIEVAL_INDEX", "/outputs/lora/iu_xray_report_lora_clean_retrieval_512/retrieval_index.json"))
    args = parser.parse_args()
    rows = []
    for line in Path(args.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method_id") == args.source_method and row.get("success"):
            rows.append(row)
    rows = rows[: args.limit]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    index = load_index(Path(args.retrieval_index))
    tokenizer, model = _get_lora_generator()
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            started = time.perf_counter()
            state = dict(row["state"])
            examples = retrieve(index, state)
            prompt = format_prompt(state, examples)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=420, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            raw_report = parse_report(generated)
            draft = _coerce_lora_report(raw_report, state)
            state["report_draft"] = draft.model_dump()
            _enforce_fewshot_checklist_report(state)
            case = EvaluationCase(case_id=row["case_id"], split=row.get("split", "test"), stratum=row.get("stratum", "unspecified"), image_paths=row.get("image_paths", []), reference_findings=row.get("reference_findings", ""), reference_impression=row.get("reference_impression", ""))
            record = record_from_state(case, "full_english_template_lora", state, time.perf_counter() - started)
            record.metadata["lora_base_model"] = os.environ.get("LORA_REPORT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
            record.metadata["lora_adapter"] = os.environ.get("LORA_REPORT_MODEL_PATH", "/outputs/lora/iu_xray_report_lora_clean_retrieval_512")
            record.metadata["retrieval_index"] = str(args.retrieval_index)
            record.metadata["retrieved_examples"] = examples
            output.write(record.model_dump_json() + "\n")
            print(f"{record.case_id} full_english_template_lora clean-retrieval: ok", flush=True)

if __name__ == "__main__":
    main()
