#!/usr/bin/env python3
"""Distribution of completion sizes per stage, so `--max-tokens` can be sized from data.

A stage whose maximum equals the cap is not "large", it is *truncated* — for the guided citation
stage that means an unparseable JSON reply, so the shape of the tail matters, not just its top.

Usage (on the A4000): .venv/bin/python3 scripts/_probe_max_completion.py <costs.jsonl> [...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

CAP = 4096

by_component: dict[str, list[int]] = defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path):
        row = json.loads(line)
        by_component[row.get("component") or "?"].append(row.get("output_tokens") or 0)

for component, tokens in sorted(by_component.items()):
    tokens.sort()
    n = len(tokens)
    at_cap = sum(1 for t in tokens if t >= CAP)
    print(f"{component}: n={n} min={tokens[0]} median={tokens[n // 2]} "
          f"p90={tokens[int(n * 0.9)]} max={tokens[-1]} at_cap={at_cap}")
    print(f"  top 10: {tokens[-10:]}")
