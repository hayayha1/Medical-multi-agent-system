import argparse
import asyncio
import json
from pathlib import Path

from app.agents.auditor import AuditorAgent
from app.config import get_settings
from app.graph import get_ollama_client
from evaluation.error_injection import aggregate_auditor_results
from evaluation.io import read_jsonl


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run the auditor on error-injection challenges.")
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    challenges = read_jsonl(args.challenges)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    existing = []
    if target.exists():
        existing = read_jsonl(target)
        completed = {str(row["challenge_id"]) for row in existing}
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    auditor = AuditorAgent(get_settings(), get_ollama_client())

    async def worker(challenge):
        async with semaphore:
            try:
                result = await auditor.run(challenge["state"])
                audit = result["audit_result"]
                row = {
                    **{key: challenge[key] for key in ("challenge_id", "case_id", "method_id", "error_type", "is_error")},
                    "flagged": not bool(audit.get("approved")),
                    "risk_level": audit.get("risk_level"),
                    "issue_codes": [item.get("code") for item in audit.get("issues", [])],
                    "error": None,
                }
            except Exception as exc:
                row = {
                    **{key: challenge[key] for key in ("challenge_id", "case_id", "method_id", "error_type", "is_error")},
                    "flagged": False,
                    "risk_level": "execution_error",
                    "issue_codes": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
        async with write_lock:
            with target.open("a", encoding="utf-8", newline="\n") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{challenge['challenge_id']}: {'flagged' if row['flagged'] else 'not_flagged'}", flush=True)

    try:
        await asyncio.gather(*(worker(item) for item in challenges if str(item["challenge_id"]) not in completed))
    finally:
        await get_ollama_client().close()
    rows = read_jsonl(target)
    summary = aggregate_auditor_results([row for row in rows if not row.get("error")])
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main_async())

