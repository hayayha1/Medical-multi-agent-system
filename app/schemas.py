from datetime import datetime, timezone
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class WorkflowStatus(StrEnum):
    CREATED = "created"
    ANALYZING = "analyzing"
    AWAITING_APPROVAL = "awaiting_approval"
    SIGNED = "signed"
    REJECTED = "rejected"
    FAILED = "failed"


class ClinicalContext(BaseModel):
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None
    chief_complaint: str | None = None
    history: list[str] = Field(default_factory=list)
    laboratory_results: dict[str, Any] = Field(default_factory=dict)


class AnalyzeStudyRequest(BaseModel):
    study_uid: str = Field(min_length=3)
    patient_id: str = Field(min_length=1, description="建议传入院内脱敏患者标识")
    modality: Literal["CR", "DX", "CT", "MR"] = "CT"
    body_part: str = "CHEST"
    dataset_case_id: str | None = Field(
        default=None,
        description="IU X-Ray病例/图像标识；为空时使用study_uid查找",
    )
    image_paths: list[str] = Field(
        default_factory=list,
        description="服务器数据集根目录内的相对图像路径",
    )
    clinical_context: ClinicalContext = Field(default_factory=ClinicalContext)


class ImageFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: f"F-{uuid4().hex[:8]}")
    finding_type: str
    location: str
    size_mm: list[float] = Field(default_factory=list)
    density: str | None = None
    margin: str | None = None
    confidence: float = Field(ge=0, le=1)
    series_uid: str | None = None
    instance_uid: str | None = None


class ImageAnalysisResult(BaseModel):
    study_quality: Literal["acceptable", "limited", "unusable"]
    quality_notes: list[str] = Field(default_factory=list)
    findings: list[ImageFinding]


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"E-{uuid4().hex[:8]}")
    title: str
    summary: str
    source: str
    version: str
    score: float = Field(ge=0, le=1)


class ReportStatement(BaseModel):
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class ReportDraft(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    clinical_history: str
    technique: str
    findings: list[ReportStatement]
    impression: list[ReportStatement]
    recommendations: list[ReportStatement] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    ai_disclaimer: str = "AI生成的报告草稿，仅供有资质医生审核，不得直接用于诊断或签发。"


class AuditIssue(BaseModel):
    code: str
    severity: Literal["low", "medium", "high"]
    message: str
    field: str | None = None
    suggestion: str | None = None


class AuditResult(BaseModel):
    approved: bool
    risk_level: Literal["low", "medium", "high"]
    issues: list[AuditIssue] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    doctor_id: str = Field(min_length=1)
    comment: str = ""
    edited_report: ReportDraft | None = None


class TaskResponse(BaseModel):
    task_id: str
    study_uid: str
    status: WorkflowStatus
    report_id: str | None = None
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
