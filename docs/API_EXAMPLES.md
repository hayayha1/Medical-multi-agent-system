# API 调用示例

## 创建演示分析任务

```powershell
$body = @{
  study_uid = "1.2.840.demo.001"
  patient_id = "DEIDENTIFIED-P001"
  modality = "CT"
  body_part = "CHEST"
  clinical_context = @{
    age = 62
    sex = "male"
    chief_complaint = "咳嗽两周"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/studies/analyze" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

响应中的 `report_id` 用于查询和审批报告。

## 医生批准

```powershell
$approval = @{
  decision = "approve"
  doctor_id = "DOCTOR-001"
  comment = "已核对原始影像，同意签发"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/reports/替换为报告ID/approval" `
  -Method Post `
  -ContentType "application/json" `
  -Body $approval
```

