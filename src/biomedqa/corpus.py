"""The ~2M distractor sample, and the gold dedup that gates the W2 encode (ADR-0012 §1).

ADR-0003 fixed the corpus *size*; ADR-0012 fixed its **source and selection policy**: a uniform,
seeded sample of `MedRAG/pubmed`, with the ID list committed and hashed into
`RunConfig.index_fingerprint()`. Uniform is the point — hand-picked hard negatives look like a
corpus engineered around the gold set, in a paper whose headline is a fairness comparison.

**The dedup is exclusion at draw time, not reconciliation afterwards.** PubMedQA contexts *are*
PubMed abstracts, so `MedRAG/pubmed` almost certainly contains them, and a naive union yields the
same abstract under two `passage_id`s — at which point `gold_rank` and hit@5 miscount **silently**.
Excluding gold PMIDs during the draw means there is never a moment when two copies exist, so
nothing has to decide which copy survives and no half-deduped index can reach the encoder.

**The gold copy that survives is PubMedQA's, not MedRAG's**, and that is not a preference:

- Citations are char spans into `Instance.abstract_text` (`CONTEXT.md`, `data.py`). MedRAG's text is
  a different string — `contents` prepends the title — so indexing its copy would invalidate every
  gold offset and every annotation record written against one.
- MedRAG carries no BACKGROUND/METHODS/RESULTS labels, which `ChunkConfig.keep_section_labels` and
  the section/sentence-window strategies are built on.
- `gold_passage_ids` is `{pubid}:{i}`, the id space `gold_rank` and hit@5 are defined over.

**Two silent failures this module makes loud.** Both are silent in the sense that matters: the run
completes, the numbers look plausible, and nothing downstream can tell.

1. **The partial parquet.** `MedRAG/pubmed`'s HF auto-converted parquet is a *partial* export —
   2,209,839 rows of 23,898,701 — and the shards are PMID-ascending, so it is the oldest ~9% of
   PubMed. Its size is within 10% of our 2M target, so "take 2M from MedRAG/pubmed" lands on a
   corpus of pre-1990 abstracts against 1990s–2010s gold, where gold is separable by era and
   vocabulary alone. G1's hit@5 would look excellent for the wrong reason and G2 would have nothing
   plausible to mis-cite — ADR-0003's fatal scenario, arriving through the one path where every
   number still looks sane. **Read `data_files="chunk/*.jsonl"`, never the bare dataset id**, and
   `draw_corpus` refuses any scan whose row count is not `MEDRAG_TOTAL_ROWS`.
2. **The str/int join.** PubMedQA's `pubid` is int32 and `data.py` stringifies it; MedRAG's `PMID`
   is int64. `{"21645374"} & {21645374}` is empty, so a broken join reports "0 duplicates removed",
   which reads as good news. `draw_corpus` raises on non-`int` keys, and raises again if a full scan
   collides with *no* gold PMID at all.

Selection is **bottom-k on a keyed hash**, not reservoir sampling. Three properties follow, and the
third is the reason it is worth the O(target_n) memory:

- exactly `target_n` rows, so `corpus_id = "pubmed-2m-v1"` is literally true;
- independent of the order shards are read in, so a resumed or parallelised scan is the same corpus;
- **nested across sizes** — the 1M draw at seed *s* is a strict subset of the 2M draw at seed *s*.
  R1's fallback is then a subset of the real corpus rather than an unrelated sample, and is built by
  truncating the ID list instead of re-scanning 54 GB.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import canonical_hash

#: Read this glob, never the bare dataset id — see the module docstring's failure 1.
MEDRAG_REPO = "MedRAG/pubmed"
MEDRAG_DATA_FILES = "chunk/*.jsonl"

#: 23,898,701 snippets over 1,166 PMID-ordered shards (the dataset card, verified 2026-08-05). Any
#: other scanned count means the glob missed shards or resolved to the partial parquet export.
MEDRAG_TOTAL_ROWS = 23_898_701

#: ADR-0003. `RetrievalConfig.corpus_id` is `"pubmed-2m-v1"`; this is the number behind that name.
TARGET_N = 2_000_000

#: The draw is reproducible from (seed, target_n) alone. Changing this changes the corpus.
CORPUS_SEED = 20260810      # W2's opening date, so the seed records when the corpus was drawn

GOLD_PMID_ARTIFACT_VERSION = "1.0.0"
GOLD_PMIDS_PATH = Path(__file__).resolve().parents[2] / "data" / "gold_pmids.json"


def selection_key(pmid: int, *, seed: int) -> int:
    """The sort key a PMID is drawn on. Uniform in `pmid`, and stable across processes.

    `hash()` would not do: Python salts it per process, so a draw keyed on it would differ between
    this machine and the box while looking perfectly reproducible on either one.
    """
    digest = hashlib.blake2b(f"{seed}:{pmid}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


#: How many standard deviations of headroom the prescan superset carries over the target. 30 makes
#: the superset failing to contain the draw a non-event; the cost is ~2% more disk.
PRESCAN_SIGMA = 30


def prescan_cutoff(target_n: int, total_rows: int) -> tuple[int, int]:
    """`(key_cutoff, expected_superset_size)` for the one-pass scan in `scripts/build_corpus.py`.

    Bottom-k does not know what it kept until the scan ends, which would force a second 54 GB read
    to fetch the drawn rows' text. Keeping every row under a slightly generous key cutoff avoids
    that: the exact bottom-k is then taken from the superset on disk.

    **Derived from `target_n`, never hard-coded.** The superset overshoots by `PRESCAN_SIGMA`
    standard deviations of a Binomial(total_rows, target_n/total_rows) count. A fixed overshoot
    looks fine at 2M and silently becomes a coin flip at a smaller `--target-n` — which is exactly
    what R1's 1M fallback would have reached for.
    """
    if not 0 < target_n <= total_rows:
        raise ValueError(f"target_n {target_n:,} out of range for {total_rows:,} rows")
    # sqrt(target_n) overstates the real sd, which is sqrt(over * (1 - rate)) — conservative on
    # purpose. Ceiling, not truncation: at target 100,000 the truncated form lands at 29.997 sd.
    sd = target_n**0.5
    over = min(total_rows, target_n + math.ceil(max(PRESCAN_SIGMA * sd, 1000)))
    return int((over / total_rows) * (2**64)), over


@dataclass(frozen=True, slots=True)
class CorpusDraw:
    """One seeded draw, and everything needed to say what produced it.

    `fingerprint` hashes the ID list itself, not the parameters — the parameters are a *claim* about
    which corpus this is, and the ID list is the corpus. That distinction is the ADR-0007 lesson:
    cardinality, or any other summary, is never evidence that an index is the index you meant.
    """

    pmids: tuple[int, ...]
    seed: int
    target_n: int
    n_scanned: int
    n_gold_collisions: int
    fingerprint: str

    def to_json(self) -> dict[str, Any]:
        return {
            "corpus_id": "pubmed-2m-v1",
            "source": {"repo": MEDRAG_REPO, "data_files": MEDRAG_DATA_FILES},
            "seed": self.seed,
            "target_n": self.target_n,
            "n_scanned": self.n_scanned,
            "n_gold_collisions": self.n_gold_collisions,
            "fingerprint": self.fingerprint,
            "pmids": list(self.pmids),
        }


def draw_corpus(
    rows: Iterable[dict[str, Any]],
    *,
    gold_pmids: set[int],
    target_n: int = TARGET_N,
    seed: int = CORPUS_SEED,
    expected_rows: int = MEDRAG_TOTAL_ROWS,
) -> CorpusDraw:
    """Draw `target_n` non-gold PMIDs uniformly from a full scan of `rows`.

    `rows` is streamed exactly once and never held in memory; only the running bottom-k is retained.
    Every guard here fires *before* the encode, because each of them is a failure that the encode
    would otherwise bake into an index and hide behind a plausible number.
    """
    if not all(isinstance(p, int) and not isinstance(p, bool) for p in gold_pmids):
        raise TypeError(
            "gold_pmids must be int, not str: PubMedQA's pubid is int32 and data.py stringifies "
            "it, while MedRAG's PMID is int64. A str/int join matches nothing and reports zero "
            "duplicates removed, which reads as success. Load them with load_gold_pmids()."
        )

    # A max-heap of the target_n smallest keys, so the largest retained key sits at heap[0] and can
    # be evicted in O(log n). Negated because heapq is a min-heap.
    heap: list[tuple[int, int]] = []
    n_scanned = 0
    n_gold_collisions = 0

    for row in rows:
        n_scanned += 1
        pmid = row["PMID"]
        if pmid in gold_pmids:
            n_gold_collisions += 1
            continue
        key = -selection_key(pmid, seed=seed)
        if len(heap) < target_n:
            heapq.heappush(heap, (key, pmid))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, pmid))

    if n_scanned != expected_rows:
        raise ValueError(
            f"scanned {n_scanned:,} rows, expected {expected_rows:,}. MedRAG/pubmed is "
            f"{MEDRAG_TOTAL_ROWS:,} rows over 1,166 PMID-ordered shards. A short scan means the "
            f"shard glob missed shards, or the read resolved to HF's partial parquet export "
            f"(2,209,839 rows — the oldest ~9% of PubMed, and within 10% of the 2M target). "
            f"Either way the draw is not uniform over PubMed. Read {MEDRAG_DATA_FILES!r}."
        )
    if n_gold_collisions == 0:
        raise ValueError(
            "a full scan collided with no gold PMID. PubMedQA contexts are PubMed abstracts, so "
            "this is a broken join, not an absent overlap — check that PMIDs are int on both "
            "sides. (Any count above zero is reported, not gated: nobody has measured what the "
            "real overlap is, and a threshold on an unmeasured quantity buys false comfort.)"
        )
    if len(heap) < target_n:
        raise ValueError(
            f"non-gold pool holds {len(heap):,} rows, short of the {target_n:,} target"
        )

    pmids = tuple(sorted(pmid for _, pmid in heap))
    return CorpusDraw(
        pmids=pmids,
        seed=seed,
        target_n=target_n,
        n_scanned=n_scanned,
        n_gold_collisions=n_gold_collisions,
        fingerprint=canonical_hash(list(pmids)),
    )


def write_gold_pmids(pmids: Iterable[int], path: Path = GOLD_PMIDS_PATH) -> dict[str, Any]:
    """Freeze the dedup key set as a committed, hashed artifact.

    Dedup then has an auditable *input* rather than being a side effect of whichever script ran.
    The hash goes into the run manifest, so a silently edited key set cannot masquerade as the one
    the corpus was drawn against.
    """
    payload: dict[str, Any] = {
        "version": GOLD_PMID_ARTIFACT_VERSION,
        "source": "qiaojin/PubMedQA:pqa_labeled",
        "pmids": sorted(int(p) for p in pmids),
    }
    payload["hash"] = canonical_hash({k: v for k, v in payload.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_gold_pmids(path: Path = GOLD_PMIDS_PATH) -> set[int]:
    """Read the frozen key set as **ints**, verifying the hash.

    Returning `set[int]` is the single point at which the str/int trap is closed — every caller gets
    the type `draw_corpus` demands, so there is no correct-looking way to pass strings.
    """
    payload = json.loads(path.read_text())
    expected = payload.pop("hash", None)
    actual = canonical_hash(payload)
    if expected != actual:
        raise ValueError(
            f"gold PMID hash mismatch: recorded {expected}, computed {actual}. The key set was "
            "edited after freezing — any corpus drawn against it is suspect."
        )
    return {int(p) for p in payload["pmids"]}


def stream_medrag(cache_dir: str | None = None) -> Iterator[dict[str, Any]]:
    """Stream all 1,166 shards of `MedRAG/pubmed`, one row at a time.

    Explicitly `data_files=` rather than the bare dataset id: the bare form can resolve to HF's
    partial parquet export, which is the module docstring's failure 1. `draw_corpus`'s row-count
    guard is the backstop, but the right fix is not to ask for the wrong thing in the first place.
    """
    from datasets import load_dataset

    ds = load_dataset(
        MEDRAG_REPO, data_files=MEDRAG_DATA_FILES, split="train",
        streaming=True, cache_dir=cache_dir,
    )
    yield from ds
