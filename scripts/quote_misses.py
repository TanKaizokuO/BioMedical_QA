#!/usr/bin/env python3
"""Classify every CITE line a records file wrote that `locate_quote` refused.

`parse_response` reports "quote not found verbatim in {pid} ({quote[:60]!r})" and stops there,
which is the right amount of detail for an error list and not enough to decide anything: a quote
that misses on one leading capital and a quote invented wholesale produce the same message. G0's
number is a rate, but the decision that follows it — is the generator's format compliance fixable
by instruction, or is it fabricating evidence — is a question about *which* mutations occur.

So: re-read the CITE lines out of `raw_generation`, diff each miss against the passage it names,
and bucket it. The buckets are ordered by how much they matter, not by frequency:

- `fabricated`   — no substantial span of the quote appears in the passage. Evidence invented.
- `spliced`      — prefix and suffix both appear, separated by material the model deleted. Reads
                   as a verbatim quote, asserts a relation the passage does not.
- `reworded`     — word order, abbreviation expansion, or digit/word normalisation of numerals.
- `overrun`      — the quote is a correct prefix that ends early, usually with an added full stop.
- `case`         — differs only in letter case, typically the first character.

The first two are why `locate_quote` is exact-match and stays that way: a fuzzy matcher scores them
as near hits and writes `char_start`/`char_end` for text the passage does not contain.

Reads any records JSONL written by the generation path; defaults to the W4 smoke run.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

#: A run of this many characters shared with the passage means the model was copying, not inventing.
_ANCHOR = 24
#: Two anchors this far apart in the passage, adjacent in the quote, is a splice rather than a slip.
_SPLICE_GAP = 12


def cite_lines(raw: str) -> list[tuple[str, str]]:
    """`(passage_id, quote)` for every CITE line, ids unbracketed. Malformed lines are skipped —
    they are a *different* G0 failure and `parse_response` already counts them."""
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s.startswith("CITE") or "||" not in s:
            continue
        lhs, quote = s.split("||", 1)
        _, _, pid = lhs.partition(":")
        out.append((pid.strip().strip("[]"), quote.strip()))
    return out


def classify(quote: str, text: str) -> str:
    """Which mutation turned a passage span into `quote`."""
    if quote.lower() in text.lower():
        return "case"
    blocks = [
        b
        for b in difflib.SequenceMatcher(None, quote, text, autojunk=False).get_matching_blocks()
        if b.size >= _ANCHOR
    ]
    if not blocks:
        return "fabricated"
    for a, b in zip(blocks, blocks[1:]):
        quote_gap = b.a - (a.a + a.size)
        text_gap = b.b - (a.b + a.size)
        if text_gap - quote_gap >= _SPLICE_GAP:
            return "spliced"
    covered = sum(b.size for b in blocks)
    if covered >= len(quote) - 2 and blocks[0].a == 0:
        return "overrun"
    return "reworded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="?", type=Path,
                    default=_REPO / "docs" / "harvest" / "generate_smoke.records.jsonl")
    ap.add_argument("--quiet", action="store_true", help="counts only, no per-miss detail")
    args = ap.parse_args()

    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    counts: Counter[str] = Counter()
    per_system: dict[str, Counter[str]] = {}

    for rec in records:
        texts = {p["passage_id"]: p["text"] for p in rec["retrieved"]}
        system = rec["system"]
        for pid, quote in cite_lines(rec.get("raw_generation", "")):
            text = texts.get(pid)
            if text is None:
                kind = "unknown_id"
            elif quote in text:
                continue
            else:
                kind = classify(quote, text)
            counts[kind] += 1
            per_system.setdefault(system, Counter())[kind] += 1
            if not args.quiet:
                print(f"[{kind}] {system} {rec['query_id']} {pid}\n    {quote!r}")

    print(f"\n{sum(counts.values())} misses over {len(records)} records: {dict(counts)}")
    for system in sorted(per_system):
        print(f"  {system}: {dict(per_system[system])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
