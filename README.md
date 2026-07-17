# 医疗影像多智能体报告系统

基于 FastAPI + LangGraph 的四智能体医疗影像报告草稿系统：

1. 影像分析 Agent
2. 知识检索 Agent
3. 主治医生 Agent
4. 合规审计 Agent

> 本项目只生成医生待审核草稿，不能自动签发医疗报告，也不能替代医生诊断。

## 快速启动

```powershell
Copy-Item .env.example .env
docker compose up --build
```

打开 `http://localhost:8000/docs` 查看接口。

无需外部模型即可使用 `APP_MODE=demo` 跑通流程。生产配置已支持目标 Ollama 服务和 IU X-Ray 数据集；生产使用前仍必须导入审核过的医学知识、配置身份认证并完成临床验证。

## 目标虚拟机部署

数据随项目保存在 `./data/iu-xray`，Compose 会只读挂载到容器内的 `/data/iu-xray`；因此整个项目文件夹可以移动到虚拟机任意目录。

```bash
cd medical-multi-agent
cp .env.example .env
# 修改 .env 中的数据库密码、APP_SECRET、医院信息和知识审核人
docker compose up -d --build
docker compose exec api python scripts/inspect_iu_xray.py
docker compose exec api python scripts/ingest_knowledge.py
```

知识文件放置在：

```text
./knowledge_base/approved_documents/
```

支持 `.txt`、`.md`、`.pdf` 和 `.docx`。只放入经过医学人员审核且许可允许使用的文档。

## 本地开发

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest
```

## 主要接口

- `POST /api/v1/studies/analyze`：创建分析任务
- `GET /api/v1/studies/{task_id}`：查询任务
- `GET /api/v1/reports/{report_id}`：读取报告草稿
- `POST /api/v1/reports/{report_id}/approval`：医生批准、修改或拒绝
- `GET /health`：健康检查

IU X-Ray 请求可以传 `dataset_case_id`，系统会在数据目录中递归查找对应的正位/侧位图像；也可以传数据根目录内的相对 `image_paths`。

## 安全边界

- 演示模式产生的内容包含 `DEMO_ONLY` 标识。
- 正式报告必须由医生审批接口完成签发。
- 日志默认只记录任务 ID，不记录姓名、身份证号或原始影像。
- 外部大模型、PACS、FHIR 地址均为显式占位配置。

## 论文评价体系

项目包含完整的报告评价流水线：RadGraph、RadCliQ、CheXbert、GREEN、医生盲评、证据引用评价、审计错误注入、消融实验和统计检验。详见 `docs/EVALUATION.md`。


## 数据与大模型来源说明 (Data & Model Sources)
本项目所使用的超大文件（已在 .gitignore 中排除）来源如下：
1. **IU X-Ray 数据集**:
   - 原始图像归档 `NLMCXR_png.tgz` 来源于 NIH Open-I 影像库。
   - 处理后的 Parquet 数据文件（`train-00000-of-00005...` 等）来源于 Hugging Face 的 [r720/iu_xray_hf](https://huggingface.co/datasets/r720/iu_xray_hf) 数据集。
2. **Qwen3-VL & MedGemma 权重文件**:
   - Qwen3-VL-32B-Instruct 权重来源于 Hugging Face [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)。
   - MedGemma 27B 权重来源于 [google/medgemma-27b](https://huggingface.co/google/medgemma-27b)。
