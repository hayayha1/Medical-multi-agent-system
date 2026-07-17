import argparse
import asyncio

from app.config import get_settings
from app.graph import get_knowledge_store, get_ollama_client
from evaluation.generation import SUPPORTED_METHODS, run_experiment


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable report-generation ablations.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", default="full")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unsupported = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unsupported:
        raise SystemExit(f"unsupported methods: {unsupported}; choose from {SUPPORTED_METHODS}")
    try:
        await run_experiment(
            args.manifest, args.output, methods, get_settings(),
            split=args.split, limit=args.limit, concurrency=args.concurrency,
        )
    finally:
        await get_ollama_client().close()
        await get_knowledge_store().close()


if __name__ == "__main__":
    asyncio.run(async_main())

