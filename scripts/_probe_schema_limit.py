#!/usr/bin/env python3
"""Find what the vLLM structured-output backend rejects about a citation schema.

The n=100 guided run died with a bare 400 on query 17224424 — which, measured over all 300 context
records, compiles the *largest* citation schema of the set (292 enum members, 41 KB). This posts
that schema to the live server, then bisects on the number of enum members per passage to find the
boundary. Throwaway diagnostic; run on the box where port 8000 is local.

Usage (on the A4000, inside the repo): .venv/bin/python3 scripts/_probe_schema_limit.py
"""

from __future__ import annotations

import json

import httpx

from biomedqa.prompts import build_citation_response_format
from biomedqa.schema import RetrievedPassage

BASE = "http://localhost:8000/v1"
MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
CONTEXTS = "docs/harvest/parity_iter1b.records.jsonl"


def post(response_format: dict | None) -> tuple[int, str]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with the JSON object."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "seed": 0,
    }
    if response_format is not None:
        body["response_format"] = response_format
    with httpx.Client(base_url=BASE.rsplit("/v1", 1)[0], timeout=180) as client:
        r = client.post("/v1/chat/completions", json=body)
    return r.status_code, r.text[:600]


def trim(rf: dict, keep: int) -> dict:
    """Same schema with each passage's quote enum capped at `keep` members."""
    rf = json.loads(json.dumps(rf))
    branches = rf["json_schema"]["schema"]["properties"]["claims"]["items"]["properties"][
        "citations"
    ]["items"]["anyOf"]
    for branch in branches:
        branch["properties"]["quote"]["enum"] = branch["properties"]["quote"]["enum"][:keep]
    rf["json_schema"]["schema"]["properties"]["claims"]["items"]["properties"]["citations"][
        "items"
    ]["anyOf"] = [b for b in branches if b["properties"]["quote"]["enum"]]
    return rf


def main() -> int:
    records = [json.loads(line) for line in open(CONTEXTS)]
    record = next(r for r in records if str(r["query_id"]).startswith("17224424"))
    passages = [RetrievedPassage(**p) for p in record["retrieved"]]
    full = build_citation_response_format(passages, 5, 3)
    assert full is not None

    branches = full["json_schema"]["schema"]["properties"]["claims"]["items"]["properties"][
        "citations"
    ]["items"]["anyOf"]
    total = sum(len(b["properties"]["quote"]["enum"]) for b in branches)
    print(f"full schema: {len(branches)} branches, {total} enum members, {len(json.dumps(full))} bytes")

    code, text = post(full)
    print(f"  full -> {code} {text}")

    print("\nbisecting on per-passage enum cap:")
    for keep in (1, 2, 4, 8, 12, 16, 20, 24, 32, 47):
        rf = trim(full, keep)
        br = rf["json_schema"]["schema"]["properties"]["claims"]["items"]["properties"]["citations"][
            "items"
        ]["anyOf"]
        tot = sum(len(b["properties"]["quote"]["enum"]) for b in br)
        code, text = post(rf)
        flag = "ok " if code == 200 else "400"
        print(f"  keep<={keep:>3}  members={tot:>4}  bytes={len(json.dumps(rf)):>6}  {flag} {text if code != 200 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
