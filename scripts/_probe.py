#!/usr/bin/env python3
"""Throwaway: run the live re-citation stage over a few queries and show, for every quote the
parser refused, what the model wrote next to the closest thing the passage actually says."""

from __future__ import annotations

import difflib
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"

from biomedqa import backends  # noqa: E402
from biomedqa.config import GenerationConfig  # noqa: E402
from biomedqa.decompose import decompose  # noqa: E402
from biomedqa.generate import MAX_CLAIMS_PER_CITE_CALL, split_stages  # noqa: E402
from biomedqa.prompts import CONTEXT_DEPTH, build_prompt, locate_quote  # noqa: E402
from biomedqa.schema import Granularity, System  # noqa: E402

sys.path.insert(0, str(_REPO / "scripts"))
from decompose_smoke import load_post_hoc  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
config = GenerationConfig(
    backend="vllm", model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    max_tokens=4096, frequency_penalty=0.5, granularity=Granularity.ATOMIC.value,
)

for src in load_post_hoc(Path("docs/harvest/parity_iter1b.records.jsonl"), N):
    answer = split_stages(src.raw_generation)[-1]
    d = decompose(answer, config, question=src.question, seed=0, run_id="probe", query_id=src.query_id)
    context = list(src.retrieved)[:CONTEXT_DEPTH]
    text_by_id = {p.passage_id: (p.text or "") for p in context}
    claims = list(d.claims)

    for i in range(0, len(claims), MAX_CLAIMS_PER_CITE_CALL):
        batch = claims[i : i + MAX_CLAIMS_PER_CITE_CALL]
        rendered = "\n".join(f"CLAIM {j}: {c.text}" for j, c in enumerate(batch, start=1))
        prompt = build_prompt(
            System.POST_HOC, src.question, context, config.max_citations,
            stage="recite", answer=rendered, depth=CONTEXT_DEPTH, claim_count=len(batch),
        )
        raw, _ = backends.complete(prompt, config, seed=0, run_id="probe", query_id=src.query_id)
        for line in raw.splitlines():
            head, sep, rest = line.strip().partition(":")
            if not sep or head.strip().upper() != "CITE":
                continue
            pid, psep, quote = rest.partition("||")
            if not psep:
                print(f"[{src.query_id}] NO-SEP  {line.strip()[:120]}")
                continue
            pid, quote = pid.strip().strip("[]").strip(), quote.strip()
            base = pid.split(":")[0]
            cands = [k for k in text_by_id if k.split(":")[0] == base]
            if pid not in text_by_id and len(cands) == 1:
                pid = cands[0]
            if pid not in text_by_id:
                print(f"[{src.query_id}] BAD-ID  {pid!r}")
                continue
            if locate_quote(quote, pid, text_by_id[pid]) is not None:
                continue
            text = text_by_id[pid]
            words = quote.split()
            best, best_r = "", 0.0
            for m in re.finditer(r"\S+", text):
                window = text[m.start() : m.start() + len(quote) + 40]
                r = difflib.SequenceMatcher(None, quote.lower(), window.lower()).ratio()
                if r > best_r:
                    best_r, best = r, window
            print(f"[{src.query_id}] MISS r={best_r:.2f}")
            print(f"    model : {quote[:160]}")
            print(f"    passage: {best[:160]}")
