# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists — it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in. In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo (most repos):

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at the root):

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_

## Accepted ADRs are not edited — with one narrow exception

**The default is a new ADR, not an edit.** ADR-0013 rejected amending ADR-0011 in place, and ADR-0014
exists as its own record rather than as an edit to ADR-0012 for the same reason. The point is that a
reader can see what was decided, when, and on what evidence — an edited ADR silently rewrites the
past, and the reasoning in a commit message is not where anyone looks.

**Supersede rather than amend when the decision changes.** New ADR, `Supersedes ADR-00NN` in the
header, and the old one keeps its text.

**The exception, used once (ADR-0014 §2, 2026-08-06):** when a *premise* inside an accepted ADR turns
out to be wrong but **the decision it supported is unchanged**, an in-place amendment is allowed —
because a new ADR whose entire content is "the second sentence of §2 was measured on one shard" is a
worse record than a note where that sentence is. Three conditions, all required:

1. **The original text stays**, with an inline pointer to the amendment. Nothing is rewritten.
2. **The amendment is a dated block** that says what was wrong, what the new evidence is, and —
   explicitly — **what did not change**.
3. **The header records that an edit happened**, and on whose instruction.

If the *decision* changes rather than a premise, this exception does not apply. Write a new ADR.
