"""Encode the ~2M MedCPT corpus into dense embeddings with checkpoint/resume.

**Runs on the A4000 box** (CUDA required for fp16 throughput). Self-contained.

Estimated time: ~1.6 h for 2M abstracts at batch 64 on an A4000 (G0 measurement).

Usage::

    uv run python scripts/encode_corpus.py \\
        --corpus data/corpus/corpus.jsonl \\
        --out data/index \\
        --title-convention empty \\
        --strategy abstract \\
        --max-chars 2000 \\
        --batch-size 64 \\
        --resume \\
        --build-bm25

Key design decisions
--------------------
- CLS token is the article embedding: ``model(**enc).last_hidden_state[:, 0, :]``
- ``--title-convention empty``  → ``tokenizer("", text, ...)``  two-segment, empty title
- ``--title-convention single`` → ``tokenizer(text, ...)``       single segment
- Passages are L2-normalised so retrieval is a pure dot product.
- Shards of 100,000 embeddings each; ``progress.json`` tracks resume state.
- The question NEVER goes in the title slot — that is the query encoder's job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: the A4000 box clones the repo and runs under uv, so src/ is on
# sys.path from pyproject.toml's pythonpath setting.  But the script may also
# be run directly, so add it defensively.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Third-party — available on the A4000 box
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from biomedqa.chunk import chunk_instance, chunk_text
from biomedqa.config import ChunkConfig
from biomedqa.corpus import passage_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARTICLE_ENCODER = "NCBI/MedCPT-Article-Encoder"
SHARD_SIZE = 100_000
MAX_LENGTH = 512          # MedCPT's training limit
PROGRESS_FILE = "progress.json"
LOG_EVERY = 10_000


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _load_model(device: torch.device) -> tuple[AutoTokenizer, AutoModel]:
    """Load the MedCPT article encoder in fp16."""
    tok = AutoTokenizer.from_pretrained(ARTICLE_ENCODER)
    mdl = AutoModel.from_pretrained(ARTICLE_ENCODER, torch_dtype=torch.float16)
    mdl = mdl.eval().to(device)
    return tok, mdl


def _encode_batch(
    texts: list[str],
    tok: AutoTokenizer,
    mdl: AutoModel,
    device: torch.device,
    title_convention: str,
) -> np.ndarray:
    """Return L2-normalised CLS embeddings, shape (len(texts), 768), float16."""
    if title_convention == "empty":
        # Two-segment: tokenizer("", text, ...) — matches MedCPT's trained pair format
        enc = tok(
            [""] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
    else:
        # Single segment: tokenizer(text, ...) — no title slot at all
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state[:, 0, :]  # CLS token

    vecs = out.cpu().to(torch.float32).numpy().astype(np.float16)
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(vecs.astype(np.float32), axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    vecs = (vecs.astype(np.float32) / norms).astype(np.float16)
    return vecs


# ---------------------------------------------------------------------------
# Corpus streaming
# ---------------------------------------------------------------------------

def _stream_corpus(corpus_path: Path, limit: int | None = None):
    """Yield raw dicts from corpus.jsonl, one per line, stopping after `limit` rows."""
    with corpus_path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)

def _iter_passages(corpus_path: Path, chunk_cfg: ChunkConfig, *, limit: int | None,
                   with_gold: bool):
    """Yield `(passage_id, text)` for everything that belongs in the index: gold, then distractors.

    **The gold abstracts are not in `corpus.jsonl` and never will be.** ADR-0012 §1 excludes gold
    PMIDs *at draw time*, so that no abstract can enter the index twice — the draw is 2M
    distractors and zero gold, verified against the manifest. The gold side therefore has to be
    chunked from PubMedQA here.

    Two things make this the only correct source for it:

    - **The id space.** `gold_rank` and hit@5 are defined over `Instance.gold_passage_ids`, which
      is `f"{pubid}:{i}"` — the shape `chunk_instance` emits. An index built from the distractor
      corpus alone contains no id in that space at all, so **hit@5 is exactly 0.00 for every
      configuration**, and it reads as a broken retriever rather than an index missing its gold.
    - **The coordinate space.** Citations are char offsets into `Instance.abstract_text`
      (ADR-0005), so the gold copy indexed must be PubMedQA's string, not MedRAG's. This is the
      same reason `corpus.py` keeps PubMedQA's copy at dedup time.

    Gold is yielded first so that a `--limit` run — an iteration aid over distractors — still
    contains every gold passage and still produces a meaningful hit@5.
    """
    if with_gold:
        from biomedqa.data import load_instances

        instances = load_instances()
        n = 0
        for inst in instances:
            for chunk in chunk_instance(inst, chunk_cfg):
                n += 1
                yield chunk.passage_id, chunk.text
        print(f"  gold: {n:,} passages from {len(instances):,} PubMedQA abstracts "
              f"(ids are '<pubid>:<i>', the space hit@5 is defined over)", flush=True)

    for row in _stream_corpus(corpus_path, limit):
        try:
            text = passage_text(row)
        except ValueError:
            row_id = row.get("id", row.get("PMID", "?"))
            print(f"  WARNING: skipping row {row_id!r}: invalid passage_text", flush=True)
            continue
        for chunk in chunk_text(text, str(row.get("id", row.get("PMID", ""))), chunk_cfg):
            yield chunk.passage_id, chunk.text


# ---------------------------------------------------------------------------
# Progress / resume
# ---------------------------------------------------------------------------

def _read_progress(out_dir: Path) -> dict:
    p = out_dir / PROGRESS_FILE
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _write_progress(out_dir: Path, payload: dict) -> None:
    (out_dir / PROGRESS_FILE).write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# BM25 index builder
# ---------------------------------------------------------------------------

def _build_bm25(passage_texts: list[str], out_dir: Path) -> None:
    try:
        import bm25s
    except ImportError:
        print("WARNING: bm25s not importable; skipping BM25 index build.", flush=True)
        return

    bm25_dir = out_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Tokenising {len(passage_texts):,} passages for BM25…", flush=True)
    t0 = time.perf_counter()
    corpus_tokens = bm25s.tokenize(passage_texts, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(bm25_dir))
    elapsed = time.perf_counter() - t0
    print(f"  BM25 index saved to {bm25_dir}  ({elapsed:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Main encode loop
# ---------------------------------------------------------------------------

def encode(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # getattr defaults keep this callable from the sweep driver and from a bare Namespace in a
    # smoke test, without every caller having to know about the sentence-window knobs.
    chunk_cfg = ChunkConfig(
        strategy=args.strategy,
        max_chars=args.max_chars,
        window_sentences=getattr(args, "window_sentences", 3),
        stride_sentences=getattr(args, "stride_sentences", 1),
    )
    title_convention: str = args.title_convention
    assert title_convention in ("empty", "single"), (
        f"--title-convention must be 'empty' or 'single', got {title_convention!r}"
    )

    # ---- resume state -------------------------------------------------------
    progress: dict = {}
    completed_shards = 0
    if args.resume:
        progress = _read_progress(out_dir)
        completed_shards = progress.get("completed_shards", 0)
        if completed_shards:
            # Every knob here is inside `RunConfig.index_fingerprint()`. Resuming with any of them
            # changed would concatenate shards encoded under two different index identities into
            # one dense.npy — an index that matches no fingerprint, retrieves plausibly, and is
            # wrong in a way no downstream number can show. Refuse rather than reconcile.
            expected = {
                "title_convention": title_convention,
                "strategy": args.strategy,
                "max_chars": args.max_chars,
                "window_sentences": chunk_cfg.window_sentences,
                "stride_sentences": chunk_cfg.stride_sentences,
                "with_gold": getattr(args, "with_gold", True),
            }
            for key, want in expected.items():
                prev = progress.get(key)
                if prev is not None and prev != want:
                    sys.exit(
                        f"ERROR: resume mismatch — progress.json has {key}={prev!r} but this run "
                        f"asks for {want!r}. That is a different index, not a continuation of this "
                        f"one. Encode it into its own directory, or delete {out_dir}/ and restart."
                    )
            skip_passages = completed_shards * SHARD_SIZE
            print(
                f"Resuming from shard {completed_shards} "
                f"(skipping first {skip_passages:,} passages).",
                flush=True,
            )

    # ---- device -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA not available; encoding on CPU — expect very slow throughput.",
              flush=True)
    else:
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    # ---- model (load once) --------------------------------------------------
    print(f"Loading {ARTICLE_ENCODER}…", flush=True)
    tok, mdl = _load_model(device)
    print("Model loaded.", flush=True)

    # ---- stream + encode ----------------------------------------------------
    wall_start = time.perf_counter()
    total_passages = 0          # across all shards including skipped
    current_shard: int = 0
    shard_vecs: list[np.ndarray] = []
    shard_ids: list[str] = []
    #: Passage text is accumulated unconditionally, not only under --build-bm25. It is written to
    #: `passage_texts.jsonl`, which `RetrievalIndex.load` reads to populate `RetrievedPassage.text`.
    #: Without it the cross-encoder reranker scores `(query, passage_id)` — the query against the
    #: literal string "pubmed23n0001_0:0" — and generation has no passage to cite. Both fail
    #: silently, in exactly the space hit@5 is measured in.
    shard_texts: list[str] = []

    # We need passage_ids and texts for the final concatenation
    all_passage_ids: list[str] = []
    all_passage_texts: list[str] = []

    # Accumulate a batch for the encoder
    batch_texts: list[str] = []
    batch_ids: list[str] = []

    def _flush_batch() -> None:
        nonlocal shard_vecs, shard_ids, shard_texts
        if not batch_texts:
            return
        vecs = _encode_batch(batch_texts, tok, mdl, device, title_convention)
        shard_vecs.append(vecs)
        shard_ids.extend(batch_ids)
        shard_texts.extend(batch_texts)
        batch_texts.clear()
        batch_ids.clear()

    def _flush_shard(shard_idx: int) -> None:
        nonlocal shard_vecs, shard_ids, shard_texts
        if not shard_ids:
            return
        shard_mat = np.concatenate(shard_vecs, axis=0)
        shard_path = out_dir / f"shard_{shard_idx:04d}.npy"
        np.save(str(shard_path), shard_mat)

        # Append to running passage-id registry
        all_passage_ids.extend(shard_ids)
        all_passage_texts.extend(shard_texts)

        elapsed = time.perf_counter() - wall_start
        print(
            f"  Shard {shard_idx:04d} saved — {len(shard_ids):,} passages "
            f"({shard_mat.shape})  total={len(all_passage_ids):,}  "
            f"wall={elapsed:.0f}s",
            flush=True,
        )
        # Update progress
        _write_progress(out_dir, {
            "completed_shards": shard_idx + 1,
            "total_passages": len(all_passage_ids),
            "title_convention": title_convention,
            "strategy": args.strategy,
            "max_chars": args.max_chars,
            "window_sentences": chunk_cfg.window_sentences,
            "stride_sentences": chunk_cfg.stride_sentences,
            "with_gold": getattr(args, "with_gold", True),
        })

        shard_vecs = []
        shard_ids = []
        shard_texts = []

    # For the skip path we still collect passage_ids inline so passage_ids.json
    # is complete at the end without re-streaming completed shards.
    passages_in_completed_shards = completed_shards * SHARD_SIZE


    print(f"\nStreaming {corpus_path} …", flush=True)

    for pid, chunk_text_str in _iter_passages(
        corpus_path,
        chunk_cfg,
        limit=getattr(args, "limit", None),
        with_gold=getattr(args, "with_gold", True),
    ):
        total_passages += 1

        # Logging
        if total_passages % LOG_EVERY == 0:
            elapsed = time.perf_counter() - wall_start
            rate = total_passages / elapsed if elapsed > 0 else 0
            remaining = (2_000_000 - total_passages) / rate if rate > 0 else float("inf")
            print(
                f"  {total_passages:>9,} passages  "
                f"shard={current_shard:04d}  "
                f"{elapsed:.0f}s elapsed  "
                f"~{remaining/3600:.1f}h remaining  "
                f"{rate:.0f} pass/s",
                flush=True,
            )

        # Skip passages that belong to already-completed shards
        if total_passages <= passages_in_completed_shards:
            all_passage_ids.append(pid)
            all_passage_texts.append(chunk_text_str)
            # Advance shard counter without encoding
            pos_in_shard = (total_passages - 1) % SHARD_SIZE
            if pos_in_shard == SHARD_SIZE - 1:
                # This was the last passage of a completed shard
                current_shard += 1
            continue

        # Active encoding path
        batch_texts.append(chunk_text_str)
        batch_ids.append(pid)

        if len(batch_texts) >= args.batch_size:
            _flush_batch()

        # How many passages are in the current active shard so far?
        pos_in_active_shard = (total_passages - passages_in_completed_shards - 1) % SHARD_SIZE
        if pos_in_active_shard == SHARD_SIZE - 1 and batch_texts:
            # batch boundary may not align with shard boundary; flush partial batch first
            _flush_batch()

        # Check if we completed a shard
        active_encoded = total_passages - passages_in_completed_shards
        if active_encoded > 0 and active_encoded % SHARD_SIZE == 0:
            _flush_batch()  # idempotent if already flushed
            _flush_shard(current_shard)
            current_shard += 1

    # Flush final partial batch + partial shard
    _flush_batch()
    if shard_ids:
        _flush_shard(current_shard)
        current_shard += 1

    total_encoded = len(all_passage_ids)
    wall_total = time.perf_counter() - wall_start

    # -------------------------------------------------------------------------
    # Concatenate shards → dense.npy
    #
    # The filename is `dense.npy`, not `embeddings.npy`, because that is what
    # `RetrievalIndex.load` reads. It looks the file up by name and leaves
    # `dense_embeddings` as None when it is absent — so a mismatched name does not raise,
    # it yields an index that loads cleanly and retrieves nothing densely. Table 1's
    # dense and RRF rows would then be silently BM25-only.
    # -------------------------------------------------------------------------
    print(f"\nConcatenating {current_shard} shards into dense.npy …", flush=True)
    shard_arrays: list[np.ndarray] = []
    for i in range(current_shard):
        sp = out_dir / f"shard_{i:04d}.npy"
        shard_arrays.append(np.load(str(sp)))
    if shard_arrays:
        embeddings = np.concatenate(shard_arrays, axis=0)
        emb_path = out_dir / "dense.npy"
        np.save(str(emb_path), embeddings)
        emb_bytes = emb_path.stat().st_size
        print(f"  dense.npy shape={embeddings.shape}  "
              f"size={emb_bytes / 1e9:.2f} GB", flush=True)
    else:
        print("  WARNING: no shard arrays to concatenate.", flush=True)
        emb_bytes = 0

    # -------------------------------------------------------------------------
    # Write passage_ids.json and passage_texts.jsonl
    # -------------------------------------------------------------------------
    ids_path = out_dir / "passage_ids.json"
    ids_path.write_text(json.dumps(all_passage_ids))
    print(f"  passage_ids.json written ({len(all_passage_ids):,} ids)", flush=True)

    texts_path = out_dir / "passage_texts.jsonl"
    with texts_path.open("w", encoding="utf-8") as fh:
        for text in all_passage_texts:
            fh.write(json.dumps(text, ensure_ascii=False) + "\n")
    print(f"  passage_texts.jsonl written ({len(all_passage_texts):,} texts)", flush=True)

    # The three artifacts index the same passages positionally; `RetrievalIndex` looks all
    # three up by the same integer. A length skew would mis-attribute every retrieved text.
    if not (len(all_passage_ids) == len(all_passage_texts) == (embeddings.shape[0] if shard_arrays else 0)):
        sys.exit(
            f"ERROR: artifact lengths disagree — ids={len(all_passage_ids):,} "
            f"texts={len(all_passage_texts):,} "
            f"embeddings={embeddings.shape[0] if shard_arrays else 0:,}. "
            "These three are indexed positionally by the same integer; do not retrieve against this."
        )

    # -------------------------------------------------------------------------
    # BM25 index (optional)
    # -------------------------------------------------------------------------
    if args.build_bm25:
        print("\nBuilding BM25 index…", flush=True)
        _build_bm25(all_passage_texts, out_dir)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ENCODING COMPLETE")
    print(f"  Total passages encoded : {total_encoded:,}")
    print(f"  Wall time              : {wall_total/3600:.2f} h  ({wall_total:.0f} s)")
    if shard_arrays:
        print(f"  Embeddings shape       : {embeddings.shape}")
        print(f"  Embeddings on disk     : {emb_bytes / 1e9:.2f} GB")
    print(f"  Title convention       : {title_convention!r}  "
          f"(goes in index fingerprint)")
    print(f"  Chunk strategy         : {args.strategy!r}  max_chars={args.max_chars}")
    print(f"  Output directory       : {out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Encode the 2M MedCPT corpus into dense embeddings (A4000 box).",
    )
    ap.add_argument(
        "--corpus",
        default="data/corpus/corpus.jsonl",
        help="Path to the 2M-row corpus.jsonl file (default: data/corpus/corpus.jsonl).",
    )
    ap.add_argument(
        "--out",
        default="data/index",
        help="Output directory for shards, dense.npy, passage_ids.json, passage_texts.jsonl "
             "(default: data/index).",
    )
    ap.add_argument(
        "--title-convention",
        dest="title_convention",
        choices=["empty", "single"],
        default="empty",
        help=(
            "How to call the tokenizer for article text. "
            "'empty' → tok('', text, ...) two-segment with empty title (default, matches "
            "MedCPT training). "
            "'single' → tok(text, ...) single-segment, no title slot."
        ),
    )
    ap.add_argument(
        "--strategy",
        choices=["abstract", "section", "sentence_window", "fixed_width"],
        default="abstract",
        help="Chunking strategy (default: abstract).",
    )
    ap.add_argument(
        "--max-chars",
        dest="max_chars",
        type=int,
        default=2000,
        help="Maximum characters per chunk (default: 2000).",
    )
    ap.add_argument(
        "--window-sentences",
        dest="window_sentences",
        type=int,
        default=3,
        help="Sentences per window, --strategy sentence_window only (default: 3).",
    )
    ap.add_argument(
        "--stride-sentences",
        dest="stride_sentences",
        type=int,
        default=1,
        help=(
            "Sentences advanced between windows, --strategy sentence_window only (default: 1). "
            "Must not exceed --window-sentences: a wider stride tiles with gaps and the sentences "
            "in them land in no chunk at all, which reads downstream as a weak retriever rather "
            "than as an index missing text. chunk.py refuses it."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Encode only the first N corpus rows. An iteration aid for the chunker sweep — a "
            "number measured against a truncated corpus is not a Table 1 row."
        ),
    )
    ap.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Encoder batch size (default: 64, from G0 throughput measurement).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from progress.json if present; skip already-completed shards.",
    )
    ap.add_argument(
        "--build-bm25",
        dest="build_bm25",
        action="store_true",
        help="Also build and save a bm25s index to <out>/bm25/.",
    )
    ap.add_argument(
        "--no-gold",
        dest="with_gold",
        action="store_false",
        help=(
            "Exclude the PubMedQA gold abstracts from the index. Gold is ON by default: "
            "ADR-0012 §1 excludes gold PMIDs at draw time, so corpus.jsonl contains zero "
            "gold and an index built without this would have hit@5 == 0.00 by construction."
        ),
    )
    ap.set_defaults(with_gold=True)
    return ap


def main() -> int:
    args = _build_parser().parse_args()
    encode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
