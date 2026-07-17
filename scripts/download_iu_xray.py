"""Download the official IU X-Ray PNG and XML report archives from NLM Open-i."""

import argparse
from pathlib import Path

import httpx


FILES = {
    "NLMCXR_png.tgz": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
    "NLMCXR_reports.tgz": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
}


def download(client: httpx.Client, url: str, target: Path) -> None:
    existing = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    with client.stream("GET", url, headers=headers) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            existing = 0
            mode = "wb"
        total = int(response.headers.get("content-length", "0")) + existing
        received = existing
        with target.open(mode) as output:
            for chunk in response.iter_bytes(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if total:
                    print(
                        f"\r{target.name}: {received / 1024**2:.1f}/"
                        f"{total / 1024**2:.1f} MiB ({received / total:.1%})",
                        end="",
                        flush=True,
                    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    # Open-i closes some corporate proxy TLS tunnels unexpectedly; connect directly.
    with httpx.Client(follow_redirects=True, timeout=120, trust_env=False) as client:
        for name, url in FILES.items():
            download(client, url, args.destination / name)


if __name__ == "__main__":
    main()
