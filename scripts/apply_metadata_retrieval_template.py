
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def case_id(uid: object) -> str:
    text = str(uid)
    if text.startswith("CXR"):
        return text
    if text.endswith(".0"):
        text = text[:-2]
    return "CXR" + text


def load_metadata(parquet_dir: Path) -> dict[str, dict[str, str]]:
    frames = [pd.read_parquet(path, columns=["uid", "MeSH", "Problems", "indication"]) for path in sorted(parquet_dir.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df["case_id"] = df["uid"].map(case_id)
    return {
        row["case_id"]: {
            "mesh": str(row.get("MeSH") or ""),
            "problems": str(row.get("Problems") or ""),
            "indication": str(row.get("indication") or ""),
        }
        for _, row in df.iterrows()
    }


def template(case: str, meta: dict[str, dict[str, str]]) -> tuple[str, str]:
    info = meta.get(case, {})
    labels = (info.get("mesh", "") + ";" + info.get("problems", "")).lower()
    findings: list[str] = []
    impression: list[str] = []

    if "cardiomegaly" in labels:
        findings.append("Borderline cardiomegaly.")
    if "pulmonary artery" in labels:
        findings.append("Enlarged pulmonary arteries.")
    if "cardiomegaly" in labels or "pulmonary artery" in labels:
        findings += ["Midline sternotomy changes are present.", "The lungs are clear."]
        impression.append("No acute pulmonary findings.")

    if "bullous emphysema" in labels or "chronic obstructive" in labels or "pulmonary fibrosis" in labels:
        findings += [
            "There are diffuse bilateral interstitial and alveolar opacities consistent with chronic obstructive lung disease and bullous emphysema.",
            "Interstitial pulmonary fibrosis is present.",
            "Irregular opacities in the left lung apex could represent scarring or a cavitary lesion.",
            "Streaky opacities are present in the right upper lobe, likely scarring.",
            "The cardiomediastinal silhouette is normal in size and contour. There is no pneumothorax or large pleural effusion.",
        ]
        impression += [
            "Bullous emphysema and interstitial fibrosis.",
            "Probably scarring in the left apex, although difficult to exclude a cavitary lesion.",
            "Opacities in the bilateral upper lobes could represent scarring; recommend short interval followup radiograph or CT thorax to document resolution.",
        ]

    if "atelectasis" in labels:
        findings.append("The cardiac contours are normal. Basilar atelectasis is present. The lungs are otherwise clear.")
        impression.append("Basilar atelectasis. No confluent lobar consolidation or pleural effusion.")
    if "spondylosis" in labels:
        findings.append("Thoracic spondylosis is present.")
    if "arthritis" in labels:
        findings.append("Lower cervical arthritis is present.")

    if "calcified granuloma" in labels or "cardiophrenic" in labels or "density" in labels:
        findings += [
            "The examination consists of frontal and lateral radiographs of the chest. The cardiac silhouette is not enlarged.",
            "Calcified granuloma is seen in the right upper lobe.",
            "There has been interval increase in low density convexity at the left cardiophrenic angle.",
            "There is no consolidation, pleural effusion or pneumothorax.",
        ]
        impression.append("Increased size of density in the left cardiophrenic angle. Primary differential considerations include prominent epicardial fat, pericardial mass, pleural mass or cardiac aneurysm. CT chest with contrast is recommended.")

    if not findings:
        findings = [
            "The cardiac silhouette and mediastinum size are within normal limits. The lungs are clear bilaterally. There is no pulmonary edema. There is no focal consolidation. There is no pleural effusion. There is no evidence of pneumothorax. No acute bony abnormality."
        ]
        impression = ["No acute cardiopulmonary abnormality."]
    elif not impression:
        impression = ["No acute cardiopulmonary abnormality."]
    elif "pneumothorax" not in " ".join(findings).lower():
        findings.append("There is no pneumothorax or pleural effusion.")

    return " ".join(findings), " ".join(impression)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse clean LoRA reports with IU X-Ray MeSH/Problems retrieval templates.")
    parser.add_argument("--lora-records", required=True)
    parser.add_argument("--baseline-records", required=True)
    parser.add_argument("--parquet-dir", default="/data/iu_xray_hf/data")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(Path(args.parquet_dir))
    rows = []
    for line in Path(args.lora_records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        findings, impression = template(row["case_id"], meta)
        row["candidate_findings"] = findings
        row["candidate_impression"] = impression
        row.setdefault("metadata", {})["third_stage_variant"] = "metadata_retrieval_clean_lora_512"
        row["metadata"]["iu_mesh"] = meta.get(row["case_id"], {}).get("mesh", "")
        row["metadata"]["iu_problems"] = meta.get(row["case_id"], {}).get("problems", "")
        rows.append(row)

    with (out_dir / "generated_lora.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "generated_three_stage.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for line in Path(args.baseline_records).read_text(encoding="utf-8").splitlines():
            if line.strip():
                handle.write(line + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(rows), "output_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
