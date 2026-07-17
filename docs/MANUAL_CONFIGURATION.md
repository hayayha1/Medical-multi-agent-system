# 必须手动配置的项目

所有运行参数集中在根目录 `.env`。请从 `.env.example` 复制后填写。

## 生产环境必须填写

1. `APP_SECRET`：系统长随机密钥。
2. `DATABASE_URL`：生产 PostgreSQL 地址、账号和密码。
3. `REDIS_URL`：生产 Redis 地址。
4. `OLLAMA_BASE_URL`：当前为 `http://172.16.98.104:11434`。
5. 四个模型字段：当前已按服务器真实标签配置。
6. `IU_XRAY_DATASET_PATH`：宿主机为 `/home/ubuntu/hdd/mwz`，容器内为 `/data/iu-xray`。
7. `DICOMWEB_BASE_URL` 及账号密码：医院 PACS/DICOMweb。
8. `FHIR_BASE_URL`、`FHIR_TOKEN`：医院临床数据服务。
9. `HOSPITAL_NAME`、`DEPARTMENT_NAME`：医院与科室。
10. `KNOWLEDGE_APPROVER`：知识库医学审核负责人。

## 需要人工提供的数据/实现

- 经脱敏和授权的医疗影像测试集。
- 经医院审核的指南、SOP 和文献内容。
- 已完成临床验证的影像检测/分割模型。
- 医院认证和医生权限对接方式。
- 报告模板、危急值规则和科室审批流程。

## 代码中的生产适配占位

- `app/agents/image_analyst.py`：已接入 Ollama MedGemma；临床生产仍应增加专用检测/分割模型。
- `app/agents/retriever.py`：已接入 BGE-M3 + pgvector；全文混合检索可作为后续增强。
- `app/repository.py`：将演示内存存储替换为 PostgreSQL Repository。
- `app/api.py`：接入医院身份认证、后台任务队列和持久化审批。
- `app/graph.py`：生产环境接入 PostgreSQL checkpointer，并将医生审批改为 LangGraph `interrupt()` 的跨进程恢复。
