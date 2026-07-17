CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS analysis_tasks (
    id UUID PRIMARY KEY,
    study_uid TEXT NOT NULL,
    patient_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS medical_reports (
    id UUID PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES analysis_tasks(id),
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    report JSONB NOT NULL,
    audit_result JSONB NOT NULL,
    signed_by TEXT,
    signed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    version TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    valid_until DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS knowledge_embedding_hnsw
ON knowledge_documents USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS audit_events (
    id BIGSERIAL PRIMARY KEY,
    task_id UUID,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
