"""PubMedQA loading, gold-context extraction, and the frozen splits.

Distilled from the retired base pipeline — see `docs/harvest/pubmedqa-loading.md` for the row shape
and for what deliberately does not carry over. Two things it got wrong that are fixed here:

1. **`" ".join(contexts)` is not done.** The base repo flattened each abstract into one string
   because its retrieval unit was the whole abstract. That join discards `context["labels"]`, which
   are the natural boundaries for the section and sentence-window chunkers, and it destroys the
   character offsets that citations are defined in (`CONTEXT.md`).
2. **Splits are membership, not iteration order.** The base repo took "the first 50 rows". Splits
   here are frozen by pubid in a checked-in JSON whose hash goes into every run manifest.

The 1,000 `pqa_labeled` abstracts are **gold contexts, not the corpus**. They are inserted into the
~2M-abstract corpus (ADR-0003); they do not constitute it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import canonical_hash

DATASET = "qiaojin/PubMedQA"
SUBSET = "pqa_labeled"
SPLITS_PATH = Path(__file__).resolve().parents[2] / "data" / "splits.json"

DEV_N = 100
TEST_N = 400
SPLIT_SEED = 20260807   # the freeze date, so the seed itself records when this became immutable


@dataclass(slots=True)
class GoldPassage:
    """One labelled section of a PubMedQA abstract, with its offset in the reconstructed abstract.

    `char_start`/`char_end` index into `Instance.abstract_text`, which is the *only* place offsets
    are meaningful. Chunking may re-cut these; citations are defined against whatever passage ids
    the chunker emits, and those ids trace back here.
    """

    passage_id: str      # f"{pubid}:{index}"
    pubid: str
    index: int
    label: str | None    # BACKGROUND | METHODS | RESULTS | ...
    text: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class Instance:
    """One PubMedQA question and the abstract its question was written from."""

    pubid: str
    question: str
    passages: list[GoldPassage] = field(default_factory=list)
    long_answer: str = ""
    final_decision: str = ""      # yes | no | maybe

    @property
    def abstract_text(self) -> str:
        """The abstract as one string — the coordinate space every offset refers to."""
        return " ".join(p.text for p in self.passages)

    @property
    def gold_passage_ids(self) -> list[str]:
        """Gold membership is a **set**, not an identity.

        The base repo could ask `pubid in retrieved_ids` because its retrieval unit was the whole
        abstract. Under chunking one abstract becomes many passages, so "did we retrieve the gold?"
        is set intersection, and `gold_rank` is the minimum rank over this set
        (`docs/harvest/gold-passage-tracking.md`).
        """
        return [p.passage_id for p in self.passages]


def load_instances(limit: int | None = None) -> list[Instance]:
    """Load `pqa_labeled` (1,000 rows, no streaming needed) with offsets preserved."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, SUBSET, split="train")
    out: list[Instance] = []
    for row in ds:
        ctx = row["context"]
        if isinstance(ctx, dict) and "contexts" in ctx:
            texts = list(ctx["contexts"])
            labels = list(ctx.get("labels") or [None] * len(texts))
        else:                                   # the fallback the base repo needed; kept
            texts, labels = [str(ctx)], [None]

        pubid = str(row["pubid"])
        passages: list[GoldPassage] = []
        cursor = 0
        for i, (text, label) in enumerate(zip(texts, labels)):
            passages.append(
                GoldPassage(
                    passage_id=f"{pubid}:{i}",
                    pubid=pubid,
                    index=i,
                    label=label,
                    text=text,
                    char_start=cursor,
                    char_end=cursor + len(text),
                )
            )
            cursor += len(text) + 1             # the single space joined in `abstract_text`

        out.append(
            Instance(
                pubid=pubid,
                question=row["question"],
                passages=passages,
                long_answer=row.get("long_answer", "") or "",
                final_decision=row.get("final_decision", "") or "",
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def freeze_splits(
    instances: list[Instance],
    dev_n: int = DEV_N,
    test_n: int = TEST_N,
    seed: int = SPLIT_SEED,
    path: Path = SPLITS_PATH,
) -> dict:
    """Draw dev and test **once**, write them to disk, and refuse to redraw.

    Every number in the paper comes from `test`, which is run late and once per system per seed.
    Redrawing it after any tuning would silently convert the test set into a dev set, so this
    function will not overwrite an existing file — delete it deliberately if you truly mean to.

    If G0 forces a smaller test set, shrink it *here and now* and report the reduced n with CIs. A
    300-question test set with confidence intervals is a real result; a 500-question set
    half-finished in November is not (§3).
    """
    if path.exists():
        raise FileExistsError(
            f"{path} exists — the splits are frozen (§3). Load them with load_splits() instead."
        )

    pubids = sorted(i.pubid for i in instances)
    if len(pubids) < dev_n + test_n:
        raise ValueError(f"need {dev_n + test_n} instances, have {len(pubids)}")

    shuffled = pubids[:]
    random.Random(seed).shuffle(shuffled)
    payload = {
        "dataset": f"{DATASET}/{SUBSET}",
        "seed": seed,
        "dev": sorted(shuffled[:dev_n]),
        "test": sorted(shuffled[dev_n : dev_n + test_n]),
    }
    payload["hash"] = canonical_hash({k: v for k, v in payload.items() if k != "hash"})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load_splits(path: Path = SPLITS_PATH) -> dict:
    """Read the frozen splits and verify the hash. The hash goes into every run manifest, so a
    silently edited split file cannot masquerade as the frozen one."""
    payload = json.loads(path.read_text())
    expected = payload.pop("hash", None)
    actual = canonical_hash(payload)
    payload["hash"] = expected
    if expected != actual:
        raise ValueError(
            f"split file hash mismatch: recorded {expected}, computed {actual}. "
            "The splits were edited after freezing — every result derived from them is suspect."
        )
    if set(payload["dev"]) & set(payload["test"]):
        raise ValueError("dev and test overlap — the test set is contaminated")
    return payload
