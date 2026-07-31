"""PubMedQA yes/no/maybe accuracy — **secondary** (C6).

**Not yet implemented.** Due W10 (Oct 5–11).

Reported second, always. The paper's claim is attribution quality (ADR-0002); accuracy exists to
show that attribution does not *cost* correctness, not to compete with SoTA PubMedQA systems. A
framing that leads with accuracy invites exactly the comparison this paper is positioned to avoid.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..schema import QueryRecord


def accuracy(records: Iterable[QueryRecord]) -> dict:
    """Accuracy with a Wilson interval, and the 3×3 confusion over yes/no/maybe."""
    raise NotImplementedError("W10")
