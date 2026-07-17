from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from evaluation.generation import (
    _coerce_lora_report,
    _enforce_fewshot_checklist_report,
    _format_lora_prompt,
    _get_lora_generator,
    _lora_payload,
    _extract_json_object,
)
from evaluation.models import record_from_state, EvaluationCase


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LoRA third-stage reports from saved full-agent states.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--source-method", default="full_english_template")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
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

    tokenizer, model = _get_lora_generator()
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            started = time.perf_counter()
            state = dict(row["state"])
            prompt = _format_lora_prompt(_lora_payload(state))
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            try:
                raw_report = json.loads(_extract_json_object(generated))
            except Exception:
                raw_report = {"findings": [generated.strip()], "impression": []}
            draft = _coerce_lora_report(raw_report, state)
            state["report_draft"] = draft.model_dump()
            _enforce_fewshot_checklist_report(state)
            case = EvaluationCase(
                case_id=row["case_id"],
                split=row.get("split", "test"),
                stratum=row.get("stratum", "unspecified"),
                image_paths=row.get("image_paths", []),
                reference_findings=row.get("reference_findings", ""),
                reference_impression=row.get("reference_impression", ""),
            )
            record = record_from_state(case, "full_english_template_lora", state, time.perf_counter() - started)
            record.metadata["lora_base_model"] = os.environ.get("LORA_REPORT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
            record.metadata["lora_adapter"] = os.environ.get("LORA_REPORT_MODEL_PATH", "/outputs/lora/iu_xray_report_lora")
            output.write(record.model_dump_json() + "\n")
            print(f"{record.case_id} full_english_template_lora: ok", flush=True)


if __name__ == "__main__":
    main()
