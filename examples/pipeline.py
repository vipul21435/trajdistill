"""Clean, select, convert and split the synthetic sample with the Python API.

    python examples/pipeline.py
"""

from __future__ import annotations

from pathlib import Path

from trajdistill.clean import CleanConfig
from trajdistill.funnel import report_text, run_pipeline
from trajdistill.readers import read_path
from trajdistill.samples import ConvertStats, build_samples
from trajdistill.select import SelectConfig
from trajdistill.split import split_records
from trajdistill.tokens import TokenCounter

HERE = Path(__file__).parent


def main() -> None:
    counter = TokenCounter()
    report = run_pipeline(
        read_path(HERE / "sample.jsonl"),
        CleanConfig(strip_ansi=True, redact=True, normalise_whitespace=True,
                    max_observation_chars=200),
        SelectConfig(test_patches="drop", loops="cut", max_steps=20, per_instance="best",
                     max_per_repo=2),
        counter,
    )
    print(report_text(report))

    stats = ConvertStats()
    samples = list(build_samples(report.kept, style="openai", granularity="step",
                                 context_budget=300, counter=counter, stats=stats))
    print("\nstep samples:", stats.to_dict())
    first = samples[0]["messages"]
    print("first sample roles:", [m["role"] for m in first])

    train, val, info = split_records(samples, by="instance", val_fraction=0.25, seed=0)
    print("split:", info.to_dict())


if __name__ == "__main__":
    main()
