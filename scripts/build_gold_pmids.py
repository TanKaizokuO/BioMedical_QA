"""Freeze `data/gold_pmids.json` — the dedup key set the 2M draw is taken against (ADR-0012 §1).

Runs anywhere `qiaojin/PubMedQA` loads; it needs no GPU and no corpus. Committed output, so this
normally runs **once**. Re-running is safe and must reproduce the same hash.

    uv run --with datasets python scripts/build_gold_pmids.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomedqa.corpus import GOLD_PMIDS_PATH, write_gold_pmids  # noqa: E402
from biomedqa.data import load_instances  # noqa: E402


def main() -> int:
    instances = load_instances()
    payload = write_gold_pmids((int(i.pubid) for i in instances), path=GOLD_PMIDS_PATH)

    n = len(payload["pmids"])
    print(f"wrote {GOLD_PMIDS_PATH}")
    print(f"  pmids : {n:,}")
    print(f"  range : {payload['pmids'][0]} .. {payload['pmids'][-1]}")
    print(f"  hash  : {payload['hash']}")
    if n != 1000:
        print(f"\n!! pqa_labeled is 1,000 rows; got {n}. This is a different set.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
