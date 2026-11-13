from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_nebius_sample.py"


def test_fetch_creates_missing_folders_and_cleans_up(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("fetch_nebius_sample", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def fake_get(params: dict) -> dict:
        if params["length"] == 1 and params["offset"] == 0:
            return {"num_rows_total": 1000}
        rows = [{"row": {"instance_id": f"o__r-{params['offset'] + i}", "eval_logs": "big",
                         "target": True}} for i in range(params["length"])]
        return {"rows": rows}

    monkeypatch.setattr(mod, "get", fake_get)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    out = tmp_path / "not" / "yet" / "sample.jsonl.gz"
    monkeypatch.setattr(sys, "argv", ["fetch", "--rows", "30", "--run", "10", "--out", str(out)])
    mod.main()
    with gzip.open(out, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == 30 and "eval_logs" not in rows[0]
    assert not (out.parent / "sample.jsonl.gz.parts").exists()
