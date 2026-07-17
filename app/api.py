from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.datasets.iu_xray import DatasetCaseNotFound, IUXRayDataset
from app.graph import build_graph
from app.repository import repository
from app.schemas import (
    AnalyzeStudyRequest,
    ApprovalRequest,
    ReportDraft,
    TaskResponse,
    WorkflowStatus,
)

router = APIRouter()


@router.post("/studies/analyze", response_model=TaskResponse, status_code=202)
async def analyze_study(request: AnalyzeStudyRequest) -> TaskResponse:
    settings = get_settings()
    task_id = str(uuid4())
    try:
        image_paths = IUXRayDataset(settings.iu_xray_dataset_path).resolve_request(
            study_uid=request.study_uid,
            dataset_case_id=request.dataset_case_id,
            image_paths=request.image_paths,
        ) if settings.app_mode == "production" else []
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    initial_state = {
        "task_id": task_id,
        "patient_id": request.patient_id,
        "study_uid": request.study_uid,
        "modality": request.modality,
        "body_part": request.body_part,
        "image_paths": image_paths,
        "clinical_context": request.clinical_context.model_dump(),
        "retry_count": 0,
        "errors": [],
        "demo_mode": settings.app_mode == "demo",
    }
    try:
        result = await build_graph().ainvoke(initial_state)
    except Exception as exc:
        await repository.save_task(task_id, {
            **initial_state, "status": WorkflowStatus.FAILED, "error": str(exc)
        })
        raise HTTPException(status_code=503, detail="分析服务暂时不可用") from exc

    report = ReportDraft.model_validate(result["report_draft"])
    task = {
        **result,
        "status": WorkflowStatus.AWAITING_APPROVAL,
        "report_id": report.report_id,
    }
    await repository.save_task(task_id, task)
    await repository.save_report(report.report_id, {
        "task_id": task_id,
        "status": WorkflowStatus.AWAITING_APPROVAL,
        "draft": report.model_dump(),
        "audit_result": result["audit_result"],
    })
    return TaskResponse(
        task_id=task_id,
        study_uid=request.study_uid,
        status=WorkflowStatus.AWAITING_APPROVAL,
        report_id=report.report_id,
        message="报告草稿已生成，等待医生审核。",
    )


@router.get("/studies/{task_id}")
async def get_task(task_id: str) -> dict:
    task = await repository.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    task.pop("patient_id", None)
    task.pop("image_paths", None)
    return task


@router.get("/reports/{report_id}")
async def get_report(report_id: str) -> dict:
    report = await repository.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report


@router.post("/reports/{report_id}/approval")
async def approve_report(report_id: str, request: ApprovalRequest) -> dict:
    report = await repository.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")
    if report["status"] != WorkflowStatus.AWAITING_APPROVAL:
        raise HTTPException(status_code=409, detail="报告当前状态不可审批")
    if request.decision == "edit" and request.edited_report is None:
        raise HTTPException(status_code=422, detail="edit 操作必须提交 edited_report")

    if request.decision == "reject":
        status = WorkflowStatus.REJECTED
    else:
        status = WorkflowStatus.SIGNED
        if request.edited_report:
            report["draft"] = request.edited_report.model_dump()

    report.update({
        "status": status,
        "doctor_decision": request.model_dump(exclude={"edited_report"}),
    })
    await repository.save_report(report_id, report)
    return {
        "report_id": report_id,
        "status": status,
        "message": "报告已由医生签发。" if status == WorkflowStatus.SIGNED else "报告已退回。",
    }
