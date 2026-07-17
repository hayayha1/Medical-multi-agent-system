# 医疗报告生成评价体系

本模块为 IU X-Ray 四智能体实验提供可复现的三级评价：自动医学指标、放射科医生盲评、系统专项评价。评价代码与在线 API 解耦，重量级模型运行在独立 `evaluation` 容器中。

## 1. 实验终点

共同主要终点：

1. 自动指标 `RadGraph F1`（越高越好）。
2. 医生盲评的每报告临床显著错误数（越低越好）。

次要终点包括 BLEU、ROUGE、BERTScore、F1CheXbert、RadCliQ、GREEN、无临床显著错误率、完全无错误率、修改时间、证据支持率和审计错误召回率。BLEU/ROUGE不得单独用于临床有效性结论。

## 2. 数据隔离

脚本按病例 UID 的 SHA-256 哈希固定划分：20%开发集、80%测试集。同一病例的正位和侧位影像始终处于同一集合。测试参考报告不得加入知识库、提示词或模型选择过程。

```bash
python scripts/prepare_evaluation_cases.py \
  --parquet-dir /data/iu_xray_hf/data \
  --image-root /data/iu-xray/images \
  --output /outputs/manifest.jsonl
```

清单包含病例ID、参考 Findings/Impression、相对影像路径、固定分组和正常/异常分层。没有 Findings 和 Impression 的病例会被排除，并记录到 `manifest.jsonl.excluded.jsonl`；论文流程图中应报告该数量和原因。

生成后必须验证：

```bash
python scripts/validate_evaluation_manifest.py --manifest /outputs/manifest.jsonl
```

## 3. 消融实验

支持五种方法：

- `direct`：单个视觉语言模型直接生成报告。
- `no_retrieval_no_audit`：影像分析 + 主治报告，不检索、不审计。
- `no_retrieval`：无知识检索，保留审计。
- `no_audit`：保留知识检索，关闭审计。
- `full`：完整四智能体工作流。

先运行小规模验证：

```bash
python scripts/run_report_experiment.py \
  --manifest /outputs/manifest.jsonl \
  --output /outputs/generated.jsonl \
  --methods direct,no_retrieval_no_audit,no_retrieval,no_audit,full \
  --split test --limit 2 --concurrency 1
```

脚本逐条追加 JSONL，并按病例和方法自动续跑。删除 `--limit` 后运行全部测试集。并发数需要根据 Ollama 模型显存实测，默认1最稳妥。

## 4. 自动指标

基础指标无需医学评价模型：token F1、ROUGE-L fallback、smoothed BLEU-4 fallback。论文指标通过 RadEval 2.2.1 统一计算：

```bash
python scripts/evaluate_reports.py \
  --records /outputs/generated.jsonl \
  --output-dir /outputs/metrics \
  --metrics bleu,rouge,bertscore,f1chexbert,radgraph,radcliq \
  --baseline-method direct --candidate-method full
```

重型临床指标冒烟时可加 `--sections combined` 只评 Combined；论文正式结果省略该参数，默认分别统计 Findings、Impression、Combined。

需要约13.5GB模型显存的 GREEN 单独启用：

```bash
python scripts/evaluate_reports.py \
  --records /outputs/generated.jsonl \
  --output-dir /outputs/metrics_green \
  --metrics green
```

输出包括病例级指标、Bootstrap 95%置信区间、正常/异常分层结果、失败率、平均/P95推理耗时和配对随机化检验结果。RadCliQ越低越好，其余上述主要指标越高越好。

## 5. 医生盲评

```bash
python scripts/create_reader_study.py \
  --records /outputs/generated.jsonl \
  --output-dir /outputs/reader_study \
  --sample-size 200 --raters RAD-01,RAD-02
```

`blinding_key.json` 仅由统计人员保管。医生填写 `reader_study_form.csv`，对虚假发现、遗漏发现、位置/侧别、严重程度、虚构比较和遗漏比较分别记录临床显著与临床不显著错误，并填写可直接使用、完整性、清晰度和修改时间。

```bash
python scripts/aggregate_reader_study.py \
  --reader-form /outputs/reader_study/reader_study_form.csv \
  --evidence-form /outputs/reader_study/evidence_review_form.csv \
  --blinding-key /outputs/reader_study/blinding_key.json \
  --output /outputs/reader_summary.json
```

输出包括ICC(2,1)、Cohen κ和线性加权κ。若存在分歧，第三名医生应在盲态下裁决；原始双评分仍应保留用于一致性报告。

## 6. 证据评价

自动结构检查计算引用存在精度、陈述引用覆盖率、知识引用比例、检索证据利用率和检索相似度。医生在 `evidence_review_form.csv` 中标记：

- `support`: `yes`、`partial` 或 `no`。
- `source_appropriate`: 来源是否适合支持该陈述。
- `clinically_requires_evidence`: 该医学陈述是否必须有证据。

主要结果为 Citation Support Precision、Full Support Rate、Source Appropriateness和Unsupported Required Claim Rate。

## 7. 审计智能体挑战集

```bash
python scripts/build_auditor_challenges.py \
  --records /outputs/generated.jsonl \
  --output /outputs/auditor_challenges.jsonl \
  --per-error-type 20 --controls 100

python scripts/run_auditor_challenge.py \
  --challenges /outputs/auditor_challenges.jsonl \
  --output /outputs/auditor_results.jsonl \
  --summary /outputs/auditor_summary.json
```

报告敏感度、特异度、Precision、F1、误报率和六类错误的分类召回率。规则注入样本应由放射科医生抽查，确认修改确实只引入目标错误。

## 8. 汇总论文结果

```bash
python scripts/generate_evaluation_report.py \
  --automatic-summary /outputs/metrics/automatic_summary.csv \
  --execution-summary /outputs/metrics/execution_summary.csv \
  --comparisons /outputs/metrics/paired_comparisons.csv \
  --reader-summary /outputs/reader_summary.json \
  --auditor-summary /outputs/auditor_summary.json \
  --output /outputs/EVALUATION_REPORT.md
```

## 9. 独立评价容器

```bash
docker compose -f docker-compose.yml -f docker-compose.eval.yml --profile evaluation \
  build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ evaluation

docker compose -f docker-compose.yml -f docker-compose.eval.yml --profile evaluation \
  run --rm evaluation python scripts/evaluate_reports.py --help
```

首次运行医学指标会下载模型权重并缓存到 `eval_model_cache`。默认使用 `https://hf-mirror.com`，可通过环境变量 `HF_ENDPOINT` 覆盖。论文中必须记录 RadEval版本、模型版本、随机种子、失败病例数量、硬件和运行日期。
