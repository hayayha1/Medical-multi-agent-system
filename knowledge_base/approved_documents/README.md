# 审核通过的医学知识文档

只在此目录放置经过医学人员审核、且许可允许本项目使用的 `.pdf`、`.docx`、`.md` 或 `.txt` 文档。

首批建议资料：

1. 医院/实验室胸片报告模板和质控规则。
2. WHO公开的胸部影像或结核筛查指南。
3. ACR Appropriateness Criteria 中与胸部X线相关的公开材料（使用前核对许可）。
4. ICD-10、LOINC、RadLex等允许使用的术语映射。

导入前必须在 `.env` 设置 `KNOWLEDGE_APPROVER`，随后执行：

```bash
docker compose exec api python scripts/ingest_knowledge.py
```

IU X-Ray原始报告属于评测参考病例，不应自动当作临床指南。建议后续单独保存为相似病例库。

