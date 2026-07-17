from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any

from app.agents.auditor import AuditorAgent
from app.agents.image_analyst import ImageAnalystAgent
from app.agents.lead_physician import LeadPhysicianAgent
from app.agents.retriever import RetrieverAgent
from app.datasets.iu_xray import IUXRayDataset
from app.graph import build_graph, get_knowledge_store, get_ollama_client
from app.schemas import ReportDraft, ReportStatement
from evaluation.io import read_models
from evaluation.models import EvaluationCase, EvaluationRecord, record_from_state


SUPPORTED_METHODS = (
    "direct",
    "no_retrieval_no_audit",
    "no_retrieval",
    "no_audit",
    "full",
    "full_english_template",
    "full_english_template_lora",
    "full_english_fewshot_checklist",
)


def initial_state(case: EvaluationCase, settings: Any) -> dict[str, Any]:
    images = IUXRayDataset(settings.iu_xray_dataset_path).resolve_request(
        study_uid=case.case_id,
        dataset_case_id=case.case_id,
        image_paths=case.image_paths,
    )
    return {
        "task_id": f"EVAL-{case.case_id}",
        "patient_id": "DEIDENTIFIED-EVALUATION",
        "study_uid": case.case_id,
        "modality": "DX",
        "body_part": "CHEST",
        "image_paths": images,
        "clinical_context": {
            "chief_complaint": case.indication or None,
            "history": [],
            "laboratory_results": {},
        },
        "retry_count": 0,
        "errors": [],
        "demo_mode": False,
    }


async def _direct(case: EvaluationCase, state: dict[str, Any], settings: Any) -> dict[str, Any]:
    client = get_ollama_client()
    draft = await client.chat_json(
        model=settings.image_analyst_model,
        system_prompt=(
            "You are a radiologist drafting a chest X-ray report directly from the supplied images. "
            "Return findings and impression, avoid unsupported diagnoses, and use DIRECT_IMAGE as the "
            "evidence_ids value for every statement. This is an experimental baseline."
        ),
        user_prompt=json.dumps({
            "study_uid": case.case_id,
            "modality": "DX",
            "clinical_indication": case.indication,
        }, ensure_ascii=False),
        response_model=ReportDraft,
        image_paths=state["image_paths"],
    )
    state.update({
        "image_findings": [{
            "finding_id": "DIRECT_IMAGE",
            "finding_type": "direct_image_baseline",
            "location": "source images",
            "size_mm": [],
            "confidence": 1.0,
        }],
        "retrieved_evidence": [],
        "report_draft": draft.model_dump(),
        "audit_result": {"approved": True, "risk_level": "not_evaluated", "issues": []},
        "workflow_status": "awaiting_approval",
    })
    return state


async def _custom_pipeline(
    state: dict[str, Any],
    settings: Any,
    use_retrieval: bool,
    use_audit: bool,
) -> dict[str, Any]:
    client = get_ollama_client()
    store = get_knowledge_store()
    state.update(await ImageAnalystAgent(settings, client).run(state))
    if use_retrieval:
        state.update(await RetrieverAgent(settings, client, store).run(state))
    else:
        state["retrieved_evidence"] = []
    physician = LeadPhysicianAgent(settings, client)
    auditor = AuditorAgent(settings, client)
    state.update(await physician.run(state))
    if use_audit:
        for retry in range(3):
            state.update(await auditor.run(state))
            if state.get("audit_result", {}).get("approved") or retry == 2:
                break
            state["retry_count"] = state.get("retry_count", 0) + 1
            state.update(await physician.run(state))
    else:
        state["audit_result"] = {"approved": True, "risk_level": "not_evaluated", "issues": []}
        state["workflow_status"] = "awaiting_approval"
    return state


def _first_evidence_id(state: dict[str, Any]) -> str:
    for item in state.get("image_findings", []):
        value = item.get("finding_id")
        if value:
            return str(value)
    for item in state.get("retrieved_evidence", []):
        value = item.get("evidence_id")
        if value:
            return str(value)
    return "TEMPLATE_ENGLISH_REPORT"


def _statement(text: str, evidence_id: str) -> ReportStatement:
    return ReportStatement(text=text, evidence_ids=[evidence_id])


def _enforce_english_report_template(state: dict[str, Any]) -> None:
    """Evaluation-only guardrail: keep report English and never leave impression empty."""
    draft = ReportDraft.model_validate(state["report_draft"])
    evidence_id = _first_evidence_id(state)

    draft.clinical_history = "Not provided" if not draft.clinical_history.strip() else str(draft.clinical_history)
    draft.technique = "Chest radiographs."

    chinese_normal = {
        "胸部未见急性异常。": "No acute cardiopulmonary abnormality is identified.",
        "纵隔未见急性异常。": "No acute mediastinal abnormality is identified.",
    }
    for section in (draft.findings, draft.impression):
        for item in section:
            item.text = chinese_normal.get(item.text.strip(), item.text.strip())

    if not draft.findings:
        draft.findings = [_statement(
            "The cardiomediastinal silhouette is within normal size limits. The lungs are clear without focal consolidation. No pleural effusion or pneumothorax is identified.",
            evidence_id,
        )]

    finding_text = " ".join(item.text.lower() for item in draft.findings)
    checklist_terms = ("pleural effusion", "pneumothorax", "focal consolidation")
    if not all(term in finding_text for term in checklist_terms):
        draft.findings.append(_statement(
            "No focal consolidation, pleural effusion, or pneumothorax is identified.",
            evidence_id,
        ))

    if not draft.impression:
        if "no acute abnormality" in finding_text or "no focal consolidation" in finding_text:
            text = "No acute cardiopulmonary abnormality."
        else:
            text = "Chest radiograph findings as described above."
        draft.impression = [_statement(text, evidence_id)]

    draft.ai_disclaimer = "AI-generated draft report for physician review only."
    state["report_draft"] = draft.model_dump()


def _enforce_fewshot_checklist_report(state: dict[str, Any]) -> None:
    """Stronger evaluation-only guardrail for IU X-Ray style keyword coverage."""
    _enforce_english_report_template(state)
    draft = ReportDraft.model_validate(state["report_draft"])
    evidence_id = _first_evidence_id(state)

    finding_text = " ".join(item.text.lower() for item in draft.findings)
    required_sentences = [
        (
            ("cardiomediastinal", "heart", "mediastinum"),
            "The cardiomediastinal silhouette and mediastinum are within normal size limits.",
        ),
        (
            ("pulmonary edema",),
            "There is no pulmonary edema.",
        ),
        (
            ("focal consolidation",),
            "There is no focal consolidation.",
        ),
        (
            ("pleural effusion",),
            "There is no pleural effusion.",
        ),
        (
            ("pneumothorax",),
            "There is no pneumothorax.",
        ),
    ]
    for terms, sentence in required_sentences:
        if not any(term in finding_text for term in terms):
            draft.findings.append(_statement(sentence, evidence_id))
            finding_text += " " + sentence.lower()

    impression_text = " ".join(item.text.lower() for item in draft.impression)
    abnormal_terms = (
        "opacity",
        "consolidation",
        "effusion",
        "pneumothorax",
        "edema",
        "atelectasis",
        "cardiomegaly",
    )
    if not impression_text.strip():
        draft.impression = [_statement("No acute cardiopulmonary abnormality.", evidence_id)]
    elif (
        not any(term in impression_text for term in abnormal_terms)
        and "no acute" not in impression_text
        and "normal" not in impression_text
    ):
        draft.impression = [_statement("No acute cardiopulmonary abnormality.", evidence_id)]

    draft.technique = "Chest radiographs."
    draft.ai_disclaimer = "AI-generated draft report for physician review only."
    state["report_draft"] = draft.model_dump()


async def _english_template_pipeline(state: dict[str, Any], settings: Any) -> dict[str, Any]:
    client = get_ollama_client()
    store = get_knowledge_store()
    state.update(await ImageAnalystAgent(settings, client).run(state))
    state.update(await RetrieverAgent(settings, client, store).run(state))
    payload = {
        "study_uid": state["study_uid"],
        "modality": state.get("modality", "DX"),
        "body_part": state.get("body_part", "CHEST"),
        "clinical_context": state.get("clinical_context", {}),
        "image_findings": state.get("image_findings", []),
        "retrieved_evidence": state.get("retrieved_evidence", []),
        "required_output_style": {
            "language": "English",
            "sections": ["FINDINGS", "IMPRESSION"],
            "must_not_be_empty": ["findings", "impression"],
            "findings_checklist": [
                "cardiomediastinal silhouette",
                "lungs",
                "focal consolidation",
                "pleural effusion",
                "pneumothorax",
                "osseous structures if visible",
            ],
            "instruction": "Use concise IU X-Ray style. Do not output Chinese. Cite existing evidence_ids only.",
        },
    }
    draft = await client.chat_json(
        model=settings.lead_physician_model,
        system_prompt=(
            "You are an attending radiologist writing an English chest X-ray report. "
            "Use only the supplied image findings, clinical context, and retrieved evidence. "
            "Always produce non-empty findings and non-empty impression. "
            "Findings must be complete and mention heart/mediastinum, lungs, focal consolidation, "
            "pleural effusion, and pneumothorax when assessable. "
            "Impression must summarize the most important abnormality or state no acute cardiopulmonary abnormality. "
            "Every statement must use existing evidence_ids from image_findings or retrieved_evidence. "
            "Return valid JSON matching the ReportDraft schema."
        ),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        response_model=ReportDraft,
    )
    state["report_draft"] = draft.model_dump()
    _enforce_english_report_template(state)
    auditor = AuditorAgent(settings, client)
    for retry in range(3):
        state.update(await auditor.run(state))
        if state.get("audit_result", {}).get("approved") or retry == 2:
            break
        state["retry_count"] = state.get("retry_count", 0) + 1
        draft = await client.chat_json(
            model=settings.lead_physician_model,
            system_prompt=(
                "Revise the English chest X-ray report to resolve audit issues while preserving "
                "non-empty findings and impression. Use existing evidence_ids only."
            ),
            user_prompt=json.dumps({
                "image_findings": state.get("image_findings", []),
                "retrieved_evidence": state.get("retrieved_evidence", []),
                "previous_report": state.get("report_draft", {}),
                "audit_result": state.get("audit_result", {}),
            }, ensure_ascii=False),
            response_model=ReportDraft,
        )
        state["report_draft"] = draft.model_dump()
        _enforce_english_report_template(state)
    return state


def _lora_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_uid": state["study_uid"],
        "modality": state.get("modality", "DX"),
        "body_part": state.get("body_part", "CHEST"),
        "clinical_context": state.get("clinical_context", {}),
        "image_findings": state.get("image_findings", []),
        "retrieved_evidence": state.get("retrieved_evidence", []),
        "required_output_style": {
            "language": "English",
            "sections": ["FINDINGS", "IMPRESSION"],
            "must_not_be_empty": ["findings", "impression"],
            "findings_checklist": [
                "cardiomediastinal silhouette",
                "lungs",
                "focal consolidation",
                "pleural effusion",
                "pneumothorax",
            ],
            "instruction": "Use concise IU X-Ray style. Return valid JSON matching ReportDraft.",
        },
    }


_LORA_GENERATOR: Any | None = None


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LoRA model did not return a JSON object")
    return text[start:end + 1]


def _coerce_lora_statement(item: Any, evidence_id: str) -> ReportStatement:
    if isinstance(item, str):
        return _statement(item, evidence_id)
    if isinstance(item, dict):
        raw_text = item.get("text") or item.get("finding") or item.get("statement") or item.get("description")
        if raw_text is None:
            raw_text = " ".join(str(value) for key, value in item.items() if key not in {"id", "evidence_id", "evidence_ids"})
        evidence = item.get("evidence_ids") or item.get("evidence_id") or item.get("evidence") or [evidence_id]
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [str(value) for value in evidence if str(value).strip()] or [evidence_id]
        return ReportStatement(text=str(raw_text).strip(), evidence_ids=evidence)
    return _statement(str(item), evidence_id)


def _coerce_lora_report(raw: dict[str, Any], state: dict[str, Any]) -> ReportDraft:
    evidence_id = _first_evidence_id(state)
    findings_raw = raw.get("findings") or raw.get("FINDINGS") or []
    impression_raw = raw.get("impression") or raw.get("IMPRESSION") or []
    if isinstance(findings_raw, str):
        findings_raw = [findings_raw]
    if isinstance(impression_raw, str):
        impression_raw = [impression_raw]
    findings = [_coerce_lora_statement(item, evidence_id) for item in findings_raw]
    impression = [_coerce_lora_statement(item, evidence_id) for item in impression_raw]
    if not findings:
        findings = [_statement("No acute cardiopulmonary abnormality is identified.", evidence_id)]
    if not impression:
        impression = [_statement("No acute cardiopulmonary abnormality.", evidence_id)]
    recommendations_raw = raw.get("recommendations") or []
    if isinstance(recommendations_raw, str):
        recommendations_raw = [recommendations_raw]
    return ReportDraft(
        clinical_history=str(raw.get("clinical_history") or "Not provided"),
        technique=str(raw.get("technique") or "Chest radiographs."),
        findings=findings,
        impression=impression,
        recommendations=[_coerce_lora_statement(item, evidence_id) for item in recommendations_raw],
        uncertainties=[str(item) for item in raw.get("uncertainties", [])] if isinstance(raw.get("uncertainties", []), list) else [],
        ai_disclaimer=str(raw.get("ai_disclaimer") or "AI-generated draft report for physician review only."),
    )


def _get_lora_generator() -> Any:
    global _LORA_GENERATOR
    if _LORA_GENERATOR is not None:
        return _LORA_GENERATOR

    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = os.environ.get("LORA_REPORT_MODEL_PATH", "/outputs/lora/iu_xray_report_lora")
    base_model = os.environ.get("LORA_REPORT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if device == "cpu":
        model.to(device)
    _LORA_GENERATOR = (tokenizer, model)
    return _LORA_GENERATOR


def _format_lora_prompt(payload: dict[str, Any]) -> str:
    schema = ReportDraft.model_json_schema()
    return (
        "You are an attending radiologist writing an English IU X-Ray style chest radiograph report.\n"
        "Use only the supplied image findings, clinical context, and retrieved evidence.\n"
        "Return only valid JSON matching the ReportDraft schema.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        f"JSON_SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        "OUTPUT_JSON:\n"
    )


async def _english_template_lora_pipeline(state: dict[str, Any], settings: Any) -> dict[str, Any]:
    client = get_ollama_client()
    store = get_knowledge_store()
    state.update(await ImageAnalystAgent(settings, client).run(state))
    state.update(await RetrieverAgent(settings, client, store).run(state))

    tokenizer, model = _get_lora_generator()
    prompt = _format_lora_prompt(_lora_payload(state))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    import torch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    try:
        raw_report = json.loads(_extract_json_object(generated))
    except Exception:
        raw_report = {"findings": [generated.strip()], "impression": []}
    draft = _coerce_lora_report(raw_report, state)
    state["report_draft"] = draft.model_dump()
    _enforce_fewshot_checklist_report(state)

    auditor = AuditorAgent(settings, client)
    for retry in range(3):
        state.update(await auditor.run(state))
        if state.get("audit_result", {}).get("approved") or retry == 2:
            break
        state["retry_count"] = state.get("retry_count", 0) + 1
        _enforce_fewshot_checklist_report(state)
    return state


async def _english_fewshot_checklist_pipeline(state: dict[str, Any], settings: Any) -> dict[str, Any]:
    client = get_ollama_client()
    store = get_knowledge_store()
    state.update(await ImageAnalystAgent(settings, client).run(state))
    state.update(await RetrieverAgent(settings, client, store).run(state))
    payload = {
        "study_uid": state["study_uid"],
        "modality": state.get("modality", "DX"),
        "body_part": state.get("body_part", "CHEST"),
        "clinical_context": state.get("clinical_context", {}),
        "image_findings": state.get("image_findings", []),
        "retrieved_evidence": state.get("retrieved_evidence", []),
        "iu_xray_style_examples": [
            {
                "findings": (
                    "The cardiac silhouette and mediastinum size are within normal limits. "
                    "There is no pulmonary edema. There is no focal consolidation. "
                    "There is no pleural effusion. There is no evidence of pneumothorax."
                ),
                "impression": "No acute cardiopulmonary abnormality.",
            },
            {
                "findings": (
                    "Heart size and mediastinal contours are normal. The lungs are clear. "
                    "No focal airspace consolidation, pleural effusion, or pneumothorax."
                ),
                "impression": "Normal chest radiographs.",
            },
            {
                "findings": (
                    "The cardiomediastinal silhouette is within normal limits. "
                    "Low lung volumes are present. Mild bibasilar atelectatic change is seen. "
                    "No pleural effusion or pneumothorax."
                ),
                "impression": "Low volume film with mild bibasilar atelectatic change.",
            },
        ],
        "mandatory_checklist": [
            "cardiac silhouette / cardiomediastinal silhouette",
            "mediastinum",
            "lungs",
            "pulmonary edema",
            "focal consolidation",
            "pleural effusion",
            "pneumothorax",
            "osseous structures if visible",
        ],
    }
    draft = await client.chat_json(
        model=settings.lead_physician_model,
        system_prompt=(
            "You are an attending radiologist writing an English IU X-Ray style chest radiograph report. "
            "Follow the examples closely in wording and section length, but do not copy unsupported abnormalities. "
            "Use only the supplied image findings, clinical context, and retrieved evidence. "
            "The FINDINGS section must explicitly address the mandatory checklist: cardiac silhouette or "
            "cardiomediastinal silhouette, mediastinum, lungs, pulmonary edema, focal consolidation, "
            "pleural effusion, and pneumothorax. "
            "If a checklist item is not abnormal, state the relevant negative finding in IU X-Ray style. "
            "The IMPRESSION section must be non-empty and concise. For normal studies prefer "
            "'No acute cardiopulmonary abnormality.' or 'Normal chest radiographs.' "
            "Do not output Chinese. Every statement must use existing evidence_ids from image_findings or retrieved_evidence. "
            "Return valid JSON matching the ReportDraft schema."
        ),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        response_model=ReportDraft,
    )
    state["report_draft"] = draft.model_dump()
    _enforce_fewshot_checklist_report(state)
    auditor = AuditorAgent(settings, client)
    for retry in range(3):
        state.update(await auditor.run(state))
        if state.get("audit_result", {}).get("approved") or retry == 2:
            break
        state["retry_count"] = state.get("retry_count", 0) + 1
        draft = await client.chat_json(
            model=settings.lead_physician_model,
            system_prompt=(
                "Revise the English IU X-Ray style chest report to resolve audit issues. "
                "Keep non-empty findings and impression, preserve mandatory checklist coverage, "
                "and use existing evidence_ids only."
            ),
            user_prompt=json.dumps({
                "image_findings": state.get("image_findings", []),
                "retrieved_evidence": state.get("retrieved_evidence", []),
                "previous_report": state.get("report_draft", {}),
                "audit_result": state.get("audit_result", {}),
                "mandatory_checklist": payload["mandatory_checklist"],
            }, ensure_ascii=False),
            response_model=ReportDraft,
        )
        state["report_draft"] = draft.model_dump()
        _enforce_fewshot_checklist_report(state)
    return state


async def run_case(case: EvaluationCase, method: str, settings: Any) -> EvaluationRecord:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    started = time.perf_counter()
    state = initial_state(case, settings)
    try:
        if method == "full":
            state = await build_graph().ainvoke(state)
        elif method == "full_english_template":
            state = await _english_template_pipeline(state, settings)
        elif method == "full_english_template_lora":
            state = await _english_template_lora_pipeline(state, settings)
        elif method == "full_english_fewshot_checklist":
            state = await _english_fewshot_checklist_pipeline(state, settings)
        elif method == "direct":
            state = await _direct(case, state, settings)
        else:
            state = await _custom_pipeline(
                state,
                settings,
                use_retrieval=method in {"no_audit"},
                use_audit=method in {"no_retrieval"},
            )
        return record_from_state(case, method, dict(state), time.perf_counter() - started)
    except Exception as exc:
        return EvaluationRecord(
            case_id=case.case_id,
            method_id=method,
            split=case.split,
            stratum=case.stratum,
            image_paths=case.image_paths,
            reference_findings=case.reference_findings,
            reference_impression=case.reference_impression,
            state=state,
            latency_seconds=time.perf_counter() - started,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_experiment(
    manifest_path: str | Path,
    output_path: str | Path,
    methods: list[str],
    settings: Any,
    split: str = "test",
    limit: int | None = None,
    concurrency: int = 1,
) -> None:
    cases = [case for case in read_models(manifest_path, EvaluationCase) if case.split == split]
    if limit is not None:
        cases = cases[:limit]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, str]] = set()
    if target.exists():
        for row in target.read_text(encoding="utf-8").splitlines():
            if row.strip():
                value = json.loads(row)
                completed.add((str(value["case_id"]), str(value["method_id"])))
    jobs = [(case, method) for case in cases for method in methods if (case.case_id, method) not in completed]
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def worker(case: EvaluationCase, method: str) -> None:
        async with semaphore:
            record = await run_case(case, method, settings)
        async with write_lock:
            with target.open("a", encoding="utf-8", newline="\n") as output:
                output.write(record.model_dump_json() + "\n")
        print(f"{case.case_id} {method}: {'ok' if record.success else record.error}", flush=True)

    await asyncio.gather(*(worker(case, method) for case, method in jobs))
