# ADR-0017 — Annotation progress is a write-mostly LAN sidecar, not a dashboard

**Status:** Accepted · **Date:** 2026-08-11 · **Decided in:** annotation-tooling grilling round 2
**Refines** ADR-0016 §4 (blinding) and §2 (shared order) · **Constrains** the W6–W8 annotation window

## Context

ADR-0016 put three annotators — a1 Shreyansh, a2 Aditya, a3 Kush — on the full ~250-claim gold set
for W6–W8, and §4 forbade a shared server on the grounds that one rater must not see another's
judgements before α is computed. The form built for it is one self-contained HTML file per rater:
`localStorage` in, JSONL out, no network.

Two facts arrived after that form was verified.

1. **Ten to sixteen hours each, over three weeks, with no progress signal.** The first evidence that
   a rater has stalled is an email that does not arrive. ADR-0016 §2's stoppability argument assumes
   we know where each pass stopped; in W7 that is a question we cannot answer without asking.
2. **All three raters and the A4000 box are on one LAN.** The premise behind "offline form or
   nothing" — that any server means hosting, exposure and a rater-visible surface — is weaker than
   it was when §4 was written.

The exposure §4 actually rules out is *reading another pass mid-flight*. Storage is not that.

## Decision

The form keeps `localStorage` as its primary store and gains a **write-mostly backup sidecar**:
`scripts/annotation_collect.py`, stdlib-only, on the A4000 box beside vLLM.

1. **Additive, never load-bearing.** Every save POSTs a snapshot; a failed POST shows `no backup —
   working offline` in the header and is dropped. The form built without `--collector-url` is
   byte-for-byte the offline form ADR-0016 accepted, and a rater on a train loses nothing.
2. **Append-only.** Each POST becomes a new timestamped file under `annotation/state/<annotator>/`,
   written to `.part` and renamed. No write can destroy an earlier one.
3. **Per-annotator token, no cross-reads.** `collector_token(annotator, seed)` is derived, not
   stored, and only that annotator's form carries it. No route joins or lists annotators, and the
   keyfile never goes on the box. This is convenience, not a lock: anyone who can read the snapshot
   directory reads everything. It is our box; we accept that rather than build auth theatre.
4. **Order-hash gated.** A snapshot from a rebuilt question order is refused with 409 rather than
   stored, so a §2 violation surfaces at the first save instead of after α.
5. **Restore is deliberate and never merges.** `Restore…` offers this browser's copy, the
   collector's copy, and a JSONL the rater exported earlier, each labelled with its completion count
   and save time. The rater picks one and it *replaces* the browser's state.
6. **`GET /state/<a>/restore` returns the furthest-along pass, not the newest.** Found in live
   testing: a cleared cache mirrors one more time, and that snapshot is empty — serving the newest
   file would hand the rater back the loss they came to undo. Completion only moves forward within a
   pass, so "most complete, newest breaks ties" identifies the copy worth keeping.
7. **The maintainer sees counts, never labels.** `scripts/annotation_status.py` prints questions
   complete, claims labelled, active time and the hours that rate projects to over the full set.
   There is no view of a judgement, because that is the thing §4 protects.
8. **No service manager.** One command starts it; the same command starts it again after a reboot.

`pyproject.toml` is unchanged: `http.server` and `urllib` are enough.

## Consequences

- A rater can change machine, or lose a browser profile, and continue — previously an unrecoverable
  loss of up to a week of unpaid careful reading.
- ADR-0016 §2's stoppability becomes observable: we know each pass's prefix length during W7, not
  after it.
- The 10–16 h estimate becomes measurable from the Sep 7 pilot, because `active_s` is per question
  and projects honestly. **No threshold here gates anything** — it is a number to read.

## Weaknesses

1. **Blinding is now a convention on a shared machine, not a property of having no machine.** The
   snapshots are plain JSON on a box we all reach. Nothing but discipline stops the maintainer
   reading a1's labels while a3 is still working, and that reading would compromise α.
2. **The A4000 is copy-paste only from the agent environment.** Every restart and log read is a
   human step, and a collector that has been down for three days is indistinguishable from three
   raters who did nothing — until you look at `saved_at`.
3. **A rater can restore the wrong copy.** The panel shows counts and times precisely because the
   choice is theirs, but a wrong choice silently discards work. Append-only means it is on disk;
   noticing is manual.
4. **It is one more thing to run during the window it must not fail in.** Mitigated only by the fact
   that its failure is survivable by design.

## Alternatives rejected

- **A real dashboard, server-rendered.** Directly violates §4, and throws away a form already
  verified in a browser.
- **Email discipline and nothing else.** The status quo this replaces: no burn-down until W8, and no
  recovery at all from a cleared cache.
- **Server-side primary store, thin-client form.** Makes the A4000 a single point of failure for
  eight weeks of human labour, for no gain over mirroring.
- **Automatic merge of divergent copies.** A merge across two partial passes can leave a claim
  labelled by nobody, and nothing surfaces it until α comes out wrong.
- **Serve the newest snapshot on restore.** The defect found in live testing; see decision 6.
