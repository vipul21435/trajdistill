"""Fetch a seeded sample of rows from nebius/SWE-agent-trajectories (CC BY 4.0, see its card).

Rows come from the Hugging Face dataset viewer API in short runs at seeded random offsets, so
the sample spreads over the whole split instead of the first few repositories. The
``eval_logs`` column is dropped to save space. Requests are paced and retried on HTTP 429;
an interrupted run resumes from the runs already saved next to the output, which
are removed once the sample is written. Nothing fetched
here is committed.

    python scripts/fetch_nebius_sample.py --rows 3000 --seed 0 --out data/nebius-sample.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
DATASET = "nebius/SWE-agent-trajectories"
KEEP = ("instance_id", "model_name", "target", "trajectory", "exit_status", "generated_patch")


def get(params: dict[str, object], tries: int = 8) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "trajdistill-fetch"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            wait = max(float(exc.headers.get("Retry-After") or 0), 10.0 * 2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            wait = 5.0 * 2 ** attempt
        print(f"retry in {wait:.0f}s", file=sys.stderr)
        time.sleep(min(wait, 300.0))
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--run", type=int, default=25, help="rows per request (at most 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = {"dataset": DATASET, "config": "default", "split": "train"}
    total = get({**base, "offset": 0, "length": 1})["num_rows_total"]
    n_runs = -(-a.rows // a.run)
    slots = (total - a.run) // a.run
    rng = random.Random(a.seed)
    offsets = sorted(s * a.run for s in rng.sample(range(slots), n_runs))
    parts = Path(a.out + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    for k, off in enumerate(offsets):
        part = parts / f"{off:07d}.jsonl"
        if part.exists():
            continue
        length = min(a.run, a.rows - k * a.run)
        data = get({**base, "offset": off, "length": length})
        lines = [json.dumps({key: item["row"].get(key) for key in KEEP}) + "\n"
                 for item in data["rows"]]
        tmp = part.with_suffix(".tmp")
        tmp.write_text("".join(lines), encoding="utf-8")
        tmp.rename(part)
        print(f"{k + 1}/{n_runs} offset {off}: {len(lines)} rows", file=sys.stderr)
        time.sleep(a.pause)
    written = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as fh:
        for off in offsets:
            for line in (parts / f"{off:07d}.jsonl").read_text(encoding="utf-8").splitlines():
                fh.write(line + "\n")
                written += 1
    for part in parts.glob("*.jsonl"):
        part.unlink()
    parts.rmdir()
    print(f"wrote {written} of {total} rows to {a.out}")


if __name__ == "__main__":
    main()
