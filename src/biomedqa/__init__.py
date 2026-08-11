"""Evidence-grounded, claim-attributable biomedical QA.

Module map — each names the table it feeds (`research_roadmap.md` §1):

    config.py     every knob; hashed into the run manifest
    data.py       PubMedQA load, gold contexts, frozen splits
    chunk.py      passage granularity                              → Table 1 (per chunker, τ)
    retrieve.py   BM25 | MedCPT | RRF | cross-encoder rerank       → Table 1
    generate.py   joint / post-hoc / vanilla behind one API        → Table 2
    backends.py   vLLM | Anthropic                                 → Table 4
    decompose.py  decontextualized atomic claims                   → Table 2 (granularity rows)
    verify.py     MiniCheck (+ AlignScore) + Opus 5 judge          → Table 3
    annotate.py   blinded task build + the offline labelling form  → G4 (Krippendorff's α)
    schema.py     THE FROZEN OUTPUT SCHEMA
    scoring/      pure functions over the schema                   → Tables 1–5
    harness.py    seed loop, cost log, run manifest, config diff
"""

__all__ = ["config", "data", "schema"]
