from copy import deepcopy
import json
from typing import Any, Protocol

from fastapi.encoders import jsonable_encoder
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings


class Repository(Protocol):
    async def save_task(self, task_id: str, value: dict[str, Any]) -> None: ...
    async def get_task(self, task_id: str) -> dict[str, Any] | None: ...
    async def save_report(self, report_id: str, value: dict[str, Any]) -> None: ...
    async def get_report(self, report_id: str) -> dict[str, Any] | None: ...
    async def close(self) -> None: ...


class InMemoryRepository:
    """Only for demo/tests. Production uses PostgreSQLRepository."""

    def __init__(self):
        self.tasks: dict[str, dict[str, Any]] = {}
        self.reports: dict[str, dict[str, Any]] = {}

    async def save_task(self, task_id: str, value: dict[str, Any]) -> None:
        self.tasks[task_id] = deepcopy(jsonable_encoder(value))

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        value = self.tasks.get(task_id)
        return deepcopy(value) if value else None

    async def save_report(self, report_id: str, value: dict[str, Any]) -> None:
        self.reports[report_id] = deepcopy(jsonable_encoder(value))

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        value = self.reports.get(report_id)
        return deepcopy(value) if value else None

    async def close(self) -> None:
        return None


class PostgreSQLRepository:
    def __init__(self, database_url: str):
        self.engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)

    async def save_task(self, task_id: str, value: dict[str, Any]) -> None:
        encoded = jsonable_encoder(value)
        statement = text("""
            INSERT INTO analysis_tasks (id, study_uid, patient_ref, status, state)
            VALUES (CAST(:id AS uuid), :study_uid, :patient_ref, :status, CAST(:state AS jsonb))
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                state = EXCLUDED.state,
                updated_at = now()
        """)
        async with self.engine.begin() as connection:
            await connection.execute(statement, {
                "id": task_id,
                "study_uid": encoded["study_uid"],
                "patient_ref": encoded.get("patient_id", "REDACTED"),
                "status": str(encoded.get("status", encoded.get("workflow_status", "created"))),
                "state": json.dumps(encoded, ensure_ascii=False),
            })

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        statement = text("SELECT state FROM analysis_tasks WHERE id = CAST(:id AS uuid)")
        async with self.engine.connect() as connection:
            row = (await connection.execute(statement, {"id": task_id})).mappings().first()
        return dict(row["state"]) if row else None

    async def save_report(self, report_id: str, value: dict[str, Any]) -> None:
        encoded = jsonable_encoder(value)
        statement = text("""
            INSERT INTO medical_reports (id, task_id, status, report, audit_result)
            VALUES (CAST(:id AS uuid), CAST(:task_id AS uuid), :status,
                    CAST(:report AS jsonb), CAST(:audit AS jsonb))
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                report = EXCLUDED.report,
                audit_result = EXCLUDED.audit_result
        """)
        status = str(encoded["status"])
        async with self.engine.begin() as connection:
            await connection.execute(statement, {
                "id": report_id,
                "task_id": encoded["task_id"],
                "status": status,
                "report": json.dumps(encoded["draft"], ensure_ascii=False),
                "audit": json.dumps(encoded.get("audit_result", {}), ensure_ascii=False),
            })
            if status == "signed":
                decision = encoded.get("doctor_decision") or {}
                await connection.execute(text("""
                    UPDATE medical_reports SET signed_by = :doctor_id, signed_at = now()
                    WHERE id = CAST(:id AS uuid)
                """), {"id": report_id, "doctor_id": decision.get("doctor_id")})

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        statement = text("""
            SELECT task_id::text, status, report, audit_result, signed_by, signed_at
            FROM medical_reports WHERE id = CAST(:id AS uuid)
        """)
        async with self.engine.connect() as connection:
            row = (await connection.execute(statement, {"id": report_id})).mappings().first()
        if not row:
            return None
        return {
            "task_id": row["task_id"],
            "status": row["status"],
            "draft": dict(row["report"]),
            "audit_result": dict(row["audit_result"]),
            "signed_by": row["signed_by"],
            "signed_at": row["signed_at"].isoformat() if row["signed_at"] else None,
        }

    async def close(self) -> None:
        await self.engine.dispose()


settings = get_settings()
repository: Repository = (
    InMemoryRepository()
    if settings.app_mode == "demo"
    else PostgreSQLRepository(settings.database_url)
)
