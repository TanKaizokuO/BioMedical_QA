"""BM25 | MedCPT dense | RRF fusion | cross-encoder rerank — **Table 1**.

Settled decisions:

- **`bm25s`, not `rank_bm25`** — the latter is borderline at 2M (§3). Pyserini is the fallback if
  needed; Java 21 is present.
- **MedCPT is asymmetric**: `MedCPT-Query-Encoder` for queries, `MedCPT-Article-Encoder` for
  passages. Using one for both is a silent quality loss, not an error.
- **Passages carry no titles, gold or distractor** (`corpus.py`, `chunk.py`). MedCPT's article
  encoder is trained on the *(title, abstract)* pair, so this puts it off-distribution — uniformly,
  which is the point. **The empty title segment needs one convention, applied to every passage:**
  `tok("", abstract)` or single-segment `tok(abstract)`. Undecided; pick it by measuring dev hit@5
  both ways at W2, and record it in the index fingerprint — it is part of the index's identity.
  **Whatever is chosen, the title slot never receives the question.** `scripts/g0_medcpt_throughput.py`
  passes `row["question"]` there as a throughput stand-in; copying that here would index the query.
- The cascade is cheap-and-high-recall first (BM25 + dense → RRF over a ~100 pool), then expensive
  and high-precision (cross-encoder over that pool). Each stage is ablatable via `RetrievalConfig`,
  because Table 1 *is* those ablations.
- **Never tune τ to pass the gate.** R2's escalation ladder ends at relaxing to hit@10 and saying so
  in the paper, reframed as conditional attribution — never at moving the threshold quietly.
- The retrieved list is stored whole (`schema.QueryRecord.retrieved`); hit@k is computed in
  `scoring/retrieval.py` at a k chosen then, not now.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path

import bm25s
import numpy as np

from .config import RetrievalConfig
from .schema import RetrievedPassage


# ---------------------------------------------------------------------------
# Lazy model caches — torch and transformers are slow to import
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _get_query_encoder(model_name: str):
    """Load and cache a HuggingFace AutoModel for query encoding."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


@functools.lru_cache(maxsize=4)
def _get_article_encoder(model_name: str):
    """Load and cache a HuggingFace AutoModel for article/passage encoding."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model


@functools.lru_cache(maxsize=4)
def _get_cross_encoder(model_name: str):
    """Load and cache a SentenceTransformers CrossEncoder."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


# ---------------------------------------------------------------------------
# RetrievalIndex
# ---------------------------------------------------------------------------

@dataclass
class RetrievalIndex:
    """Holds prebuilt index artifacts for BM25 and dense retrieval.

    Index dir layout
    ----------------
    ``bm25/``           — bm25s native save directory
    ``dense.npy``       — float16 array of shape (N, 768), L2-normalised
    ``passage_ids.json`` — ordered list of passage_id strings
    ``passage_texts.jsonl`` — one JSON string per line, aligned with passage_ids
    """

    bm25_model: bm25s.BM25 | None = None
    dense_embeddings: np.ndarray | None = None   # (N, 768) float16, L2-normalised
    passage_ids: list[str] = field(default_factory=list)
    passage_texts: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_dir: Path) -> None:
        """Serialise all index artifacts under *index_dir*."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # passage ids
        (index_dir / "passage_ids.json").write_text(
            json.dumps(self.passage_ids, ensure_ascii=False), encoding="utf-8"
        )

        # passage texts (one JSON-encoded string per line for safe unicode)
        with open(index_dir / "passage_texts.jsonl", "w", encoding="utf-8") as fh:
            for text in self.passage_texts:
                fh.write(json.dumps(text, ensure_ascii=False) + "\n")

        # BM25
        if self.bm25_model is not None:
            bm25_dir = index_dir / "bm25"
            bm25_dir.mkdir(exist_ok=True)
            self.bm25_model.save(str(bm25_dir))

        # dense embeddings
        if self.dense_embeddings is not None:
            np.save(index_dir / "dense.npy", self.dense_embeddings)

    @classmethod
    def load(cls, index_dir: Path, config: RetrievalConfig) -> "RetrievalIndex":
        """Load index artifacts from *index_dir* according to *config* flags."""
        index_dir = Path(index_dir)

        # passage ids (always required)
        passage_ids: list[str] = json.loads(
            (index_dir / "passage_ids.json").read_text(encoding="utf-8")
        )

        # passage texts
        passage_texts: list[str] = []
        texts_path = index_dir / "passage_texts.jsonl"
        if texts_path.exists():
            with open(texts_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        passage_texts.append(json.loads(line))

        # BM25
        bm25_model: bm25s.BM25 | None = None
        if config.bm25:
            bm25_dir = index_dir / "bm25"
            if bm25_dir.exists():
                bm25_model = bm25s.BM25.load(str(bm25_dir), load_corpus=False)

        # dense embeddings
        dense_embeddings: np.ndarray | None = None
        if config.dense:
            dense_path = index_dir / "dense.npy"
            if dense_path.exists():
                dense_embeddings = np.load(dense_path)  # float16, (N, 768)

        return cls(
            bm25_model=bm25_model,
            dense_embeddings=dense_embeddings,
            passage_ids=passage_ids,
            passage_texts=passage_texts,
        )


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _bm25_retrieve(
    query: str,
    index: RetrievalIndex,
    pool_size: int,
) -> list[RetrievedPassage]:
    """BM25 retrieval using bm25s."""
    if index.bm25_model is None:
        raise ValueError("BM25 index not loaded (bm25_model is None)")

    # show_progress=False: these fire once per query, and a 100-question eval turns the log into
    # 400 empty progress bars that bury the actual results.
    query_tokens = bm25s.tokenize(query, show_progress=False)
    # retrieve returns (results, scores); results contains corpus indices
    results, scores = index.bm25_model.retrieve(
        query_tokens,
        k=min(pool_size, len(index.passage_ids)),
        show_progress=False,
    )

    passages: list[RetrievedPassage] = []
    # results shape: (1, k) when single query
    result_row = results[0]
    score_row = scores[0]
    for rank, (doc_idx, score) in enumerate(zip(result_row, score_row), start=1):
        pid = index.passage_ids[int(doc_idx)]
        text = index.passage_texts[int(doc_idx)] if index.passage_texts else None
        passages.append(RetrievedPassage(
            passage_id=pid,
            rank=rank,
            score=float(score),
            retriever="bm25",
            text=text,
        ))
    return passages


def _encode_query(query: str, model_name: str) -> np.ndarray:
    """Encode a single query string to a unit-normalised float32 vector."""
    import torch

    tokenizer, model = _get_query_encoder(model_name)
    with torch.no_grad():
        encoded = tokenizer(
            query,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        if next(model.parameters()).is_cuda:
            encoded = {k: v.cuda() for k, v in encoded.items()}
        output = model(**encoded)
        # MedCPT uses CLS token
        vec = output.last_hidden_state[:, 0, :].cpu().float().numpy()  # (1, 768)

    # L2 normalise
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = vec / np.where(norm == 0, 1.0, norm)
    return vec[0]  # (768,)


#: Rows of the dense matrix promoted to float32 at a time. Bounds the scratch
#: allocation at ~600 MB regardless of index size; see ``_dense_scores``.
DENSE_CHUNK_ROWS = 200_000


def _dense_scores(emb: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """Dot every float16 passage embedding against *query_vec* in float32.

    Promoting the whole matrix at once would allocate a second copy the size of
    the index — 6.6 GB at N=2.16M — on *every* query, so promote a slice at a
    time. BLAS throughput is unchanged; peak scratch is bounded.

    Chunking changes the ``sgemv`` accumulation order, which perturbs scores by
    up to one float32 ULP (~5e-8) relative to a whole-matrix cast. Measured on a
    real 1,200-passage index: 33/1200 rows differ, ranking identical. The chunk
    size is a fixed constant, so results stay reproducible run to run.
    """
    sims = np.empty(emb.shape[0], dtype=np.float32)
    for start in range(0, emb.shape[0], DENSE_CHUNK_ROWS):
        stop = min(start + DENSE_CHUNK_ROWS, emb.shape[0])
        sims[start:stop] = emb[start:stop].astype(np.float32) @ query_vec
    return sims


def _dense_retrieve(
    query: str,
    index: RetrievalIndex,
    config: RetrievalConfig,
    pool_size: int,
) -> list[RetrievedPassage]:
    """Dense retrieval via dot product against L2-normalised passage embeddings."""
    if index.dense_embeddings is None:
        raise ValueError("Dense index not loaded (dense_embeddings is None)")

    query_vec = _encode_query(query, config.query_encoder)  # (768,) float32

    sims = _dense_scores(index.dense_embeddings, query_vec)

    k = min(pool_size, len(index.passage_ids))
    top_indices = np.argpartition(sims, -k)[-k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    passages: list[RetrievedPassage] = []
    for rank, idx in enumerate(top_indices, start=1):
        pid = index.passage_ids[int(idx)]
        text = index.passage_texts[int(idx)] if index.passage_texts else None
        passages.append(RetrievedPassage(
            passage_id=pid,
            rank=rank,
            score=float(sims[int(idx)]),
            retriever="dense",
            text=text,
        ))
    return passages


def _rrf_fuse(
    lists: list[list[RetrievedPassage]],
    rrf_k: int,
) -> list[RetrievedPassage]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    score(d) = Σ  1 / (k + rank_in_list_i)   for each list i that contains d
    """
    # Accumulate RRF scores per passage_id; also track text for output
    rrf_scores: dict[str, float] = {}
    id_to_text: dict[str, str | None] = {}

    for ranked_list in lists:
        for passage in ranked_list:
            pid = passage.passage_id
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + passage.rank)
            if pid not in id_to_text:
                id_to_text[pid] = passage.text

    sorted_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)

    return [
        RetrievedPassage(
            passage_id=pid,
            rank=rank,
            score=rrf_scores[pid],
            retriever="rrf",
            text=id_to_text[pid],
        )
        for rank, pid in enumerate(sorted_ids, start=1)
    ]


def _rerank(
    query: str,
    passages: list[RetrievedPassage],
    reranker_model: str,
) -> list[RetrievedPassage]:
    """Cross-encoder reranking with MedCPT-Cross-Encoder."""
    if not passages:
        return passages

    cross_encoder = _get_cross_encoder(reranker_model)
    pairs = [(query, p.text or p.passage_id) for p in passages]
    scores = cross_encoder.predict(pairs)  # numpy array or list of floats

    scored = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)

    return [
        RetrievedPassage(
            passage_id=p.passage_id,
            rank=rank,
            score=float(s),
            retriever="rerank",
            text=p.text,
        )
        for rank, (s, p) in enumerate(scored, start=1)
    ]


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    config: RetrievalConfig,
    index: RetrievalIndex,
) -> list[RetrievedPassage]:
    """Return the ranked passage list for one query at the stages enabled in *config*.

    Cascade
    -------
    1. BM25 retrieval           (if config.bm25)
    2. Dense retrieval          (if config.dense)
    3. RRF fusion               (if both above AND config.rrf; else concatenation)
    4. Cross-encoder rerank     (if config.rerank)
    5. Truncate to config.top_k
    """
    pool: list[RetrievedPassage] = []
    ranked_lists: list[list[RetrievedPassage]] = []

    if config.bm25:
        bm25_results = _bm25_retrieve(query, index, config.pool_size)
        ranked_lists.append(bm25_results)

    if config.dense:
        dense_results = _dense_retrieve(query, index, config, config.pool_size)
        ranked_lists.append(dense_results)

    if len(ranked_lists) >= 2 and config.rrf:
        pool = _rrf_fuse(ranked_lists, config.rrf_k)
    elif ranked_lists:
        # Single source or RRF disabled: concatenate and deduplicate by passage_id
        seen: set[str] = set()
        for lst in ranked_lists:
            for p in lst:
                if p.passage_id not in seen:
                    seen.add(p.passage_id)
                    pool.append(p)
        # Re-rank by original score descending, then reassign ranks
        pool.sort(key=lambda p: p.score, reverse=True)
        pool = [
            RetrievedPassage(
                passage_id=p.passage_id,
                rank=rank,
                score=p.score,
                retriever=p.retriever,
                text=p.text,
            )
            for rank, p in enumerate(pool, start=1)
        ]

    # Trim pool before reranking (the expensive stage)
    pool = pool[: config.pool_size]

    if config.rerank and pool:
        pool = _rerank(query, pool, config.reranker)

    return pool[: config.top_k]


# ---------------------------------------------------------------------------
# Offline index-building helpers (run on GPU box)
# ---------------------------------------------------------------------------

def build_bm25_index(
    passage_ids: list[str],
    passage_texts: list[str],
) -> bm25s.BM25:
    """Build a BM25 index from a list of (id, text) pairs.

    Caller is responsible for saving via ``RetrievalIndex.save()``.
    """
    corpus_tokens = bm25s.tokenize(passage_texts, show_progress=True)
    model = bm25s.BM25()
    model.index(corpus_tokens)
    return model


def build_dense_index(
    passages: list[tuple[str, str]],
    config: RetrievalConfig,
    *,
    batch_size: int = 32,
    empty_title: bool = True,
) -> np.ndarray:
    """Encode passages with MedCPT-Article-Encoder; return L2-normalised float16 embeddings.

    Parameters
    ----------
    passages:
        List of ``(passage_id, text)`` pairs.  passage_id is unused here but
        kept for call-site symmetry with ``RetrievalIndex.passage_ids``.
    config:
        Supplies ``dense_encoder`` (``NCBI/MedCPT-Article-Encoder`` by default).
    batch_size:
        Encoding mini-batch size — tune to GPU VRAM.
    empty_title:
        If True (default), encode as ``tok("", abstract)`` (empty title segment).
        If False, encode as single-segment ``tok(abstract)``.
        Record which was used in the index fingerprint; the choice is part of the
        index's identity.  **Never pass the question as the title.**

    Returns
    -------
    numpy.ndarray of shape ``(N, 768)``, float16, L2-normalised.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = config.dense_encoder
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    texts = [text for _, text in passages]
    all_vecs: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        if empty_title:
            # Encode as (title, abstract) pair with empty title
            encoded = tokenizer(
                [""] * len(batch),
                batch,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Single-segment encoding
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
            vecs = output.last_hidden_state[:, 0, :].cpu().float().numpy()  # (B, 768)

        # L2 normalise per vector
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1.0, norms)
        all_vecs.append(vecs.astype(np.float16))

    return np.concatenate(all_vecs, axis=0)  # (N, 768) float16
