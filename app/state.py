from typing import Any, TypedDict


class MedicalReportState(TypedDict, total=False):
    task_id: str
    patient_id: str
    study_uid: str
    modality: str
    body_part: str
    image_paths: list[str]
    clinical_context: dict[str, Any]
    image_findings: list[dict[str, Any]]
    retrieved_evidence: list[dict[str, Any]]
    report_draft: dict[str, Any]
    audit_result: dict[str, Any]
    retry_count: int
    workflow_status: str
    doctor_decision: dict[str, Any] | None
    errors: list[str]
    demo_mode: bool
