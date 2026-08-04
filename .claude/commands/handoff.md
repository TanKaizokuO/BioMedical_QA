---
description: Write HANDOFF.md — the state a fresh session needs to resume this project
---

Update `HANDOFF.md` at the repo root so a new session with an empty context window can resume
work without re-deriving the project state.

Before writing, gather the facts rather than recalling them:

- `git log --oneline -8` and `git status --short`
- The current gate and its date from `research_roadmap.md` §0 and §5
- Open issues: `gh issue list --state open`
- Any ADR added since the last handoff (`ls docs/adr/`)

Then rewrite `HANDOFF.md` in full — do not append. It is a snapshot, not a log; a stale line in it
is worse than a missing one, because the next session will trust it.

It must answer, in this order:

1. **Where the project is** — current gate, its date, and whether it is passing, blocked, or unstarted.
2. **What is blocking right now** — the single next action, and who has to take it (user vs. agent).
   If the blocker is environmental, say what was actually observed, not what was inferred.
3. **What changed since the last handoff** — commits and decisions, with the reasoning that is not
   recoverable from the diff.
4. **What to read** — the shortest ordered list of files that reconstructs context. Not every file.
5. **Standing constraints** — the rules that are easy to violate by accident (least-processed-value,
   Wilson not Wald, vLLM is a network boundary, the base pipeline is retired).
6. **Traps** — things that have already gone wrong once and would go wrong again.

Write dated facts as absolute dates. Where a number is unmeasured, say "unmeasured" — never carry a
placeholder that could be mistaken for a result.
