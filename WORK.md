# Role

You are the **lead software-engineering orchestrator and final reviewer**.

Your goal is to resolve the following codebase issues while using **the minimum possible Claude Opus 5 tokens**.

You have access to:

- **Gemini 3.1 Pro** — use for difficult reasoning, architecture, cross-file changes, data integrity, and correctness-sensitive implementation.
- **Gemini 3.6 Flash** — use for straightforward implementation, tests, mechanical refactors, UX fixes, and hygiene.
- **Claude Opus 5** — YOU. Use yourself only for orchestration, ambiguous architectural decisions, conflict resolution, and final verification.

Do **not** personally perform work that can safely be delegated.

---

# Core Strategy

Use this hierarchy:

### Gemini 3.6 Flash

Delegate:

- Small isolated fixes
- Simple React/TypeScript changes
- Tests
- UI improvements
- Error handling with obvious behavior
- Cleanup
- ESLint/Prettier configuration
- TypeScript configuration
- Copy changes
- Straightforward performance optimizations

### Gemini 3.1 Pro

Delegate:

- Database migration correctness
- Data-loss prevention
- Migration architecture
- Cross-file state/cache consistency
- Reflow capacity correctness
- Async lifecycle correctness where behavior is subtle
- Changes involving multiple interacting modules
- Any fix where incorrect implementation could silently corrupt state

### Claude Opus 5

Only handle:

1. Initial decomposition and delegation.
2. Architectural decisions that Gemini cannot resolve confidently.
3. Reviewing Gemini outputs.
4. Resolving conflicts between agents.
5. Final integration verification.
6. Running/inspecting the final test suite and determining whether the work is actually complete.

Do NOT rewrite Gemini's implementation merely because you would have written it differently.

---

# Important Operating Rule

**Do not give the entire issue list to every agent.**

Each delegated agent should receive only:

- Relevant issue(s)
- Relevant files
- Necessary surrounding context
- Acceptance criteria
- Required tests

This minimizes token usage.

Do not ask an agent to "review the whole codebase" unless the issue genuinely requires it.

---

# Task

Resolve the following issues, prioritizing correctness and data safety.

## Severity 1 — CRITICAL

### #1 Database migration data loss

`scripts/migrate.ts:19`
`supabase/migrations/0003_drop_lesson_title.sql`

Problem:

- Migrator replays every `.sql` file on every run.
- `0003` drops `lessons.title`.
- `0004` recreates it empty.
- Re-running migration silently destroys lesson titles.
- Existing header comment incorrectly claims idempotency.

Required outcome:

- Introduce `schema_migrations(filename)` or an equivalent migration-tracking mechanism.
- Applied migrations must never execute again.
- Preserve existing data.
- Handle partially applied/failed migrations safely.
- Evaluate whether 0003/0004 should be squashed.
- Add tests covering repeated migration execution.

**Delegate to Gemini 3.1 Pro.**

Do not accept a solution that merely adds `IF EXISTS`/`IF NOT EXISTS` while still allowing destructive migrations to replay.

---

### #2 Playwright fresh-clone crash

`playwright.config.ts:7`

Problem:

- Top-level code throws when `e2e/.auth/session.json` does not exist.
- This prevents Playwright from starting.
- `e2e:login` cannot create the file because Playwright crashes first.

Required outcome:

- Fresh clone must be able to run the login/bootstrap flow.
- Move validation into an appropriate fixture/global setup, or automatically establish authentication.
- Existing authenticated E2E runs must continue working.
- Add/adjust tests where practical.

**Delegate to Gemini 3.6 Flash unless investigation reveals architectural complexity.**

---

### #3 localStorage failure can lose queue state

`src/campaign/useWriteQueue.ts:36`

Problem:

- `localStorage.setItem` is unguarded.
- QuotaExceededError/private-mode/storage failures can throw during editing.
- Queue state can be lost and React tree can unmount.

Required outcome:

- Catch storage failures.
- Preserve the queue in memory.
- Never crash the editing UI because persistence failed.
- Clearly distinguish persistence failure from successful persistence.

**Delegate to Gemini 3.1 Pro.**

---

### #4 Reflow exceeds physical capacity

`src/domain/reflowEngine.ts:177`

Problem:

- `room()` uses `plannedCapacity`.
- It ignores `target.ceiling`.
- Reflow can therefore generate physically impossible schedules exceeding:

`MAX_PROBLEMS_PER_DAY * daysLeft`

Required outcome:

- Capacity must respect both planned capacity and target ceiling.
- Correct formula should effectively constrain available room using:

`Math.min(capacity, ceiling) - unsolved - placed`

- Add regression tests for the overflow case.
- Verify adjacent reflow logic does not introduce an off-by-one error.

**Delegate to Gemini 3.1 Pro.**

---

# Severity 2 — CORRECTNESS

### #5 Queue drain stuck state

`src/campaign/useWriteQueue.ts:50`

Add:

- bounded exponential backoff
- retry limit
- explicit failure state/UI signal
- queue must not remain silently pending forever

**Gemini 3.1 Pro** because this interacts with #3.

---

### #6 Reflow cache inconsistency

`src/campaign/useCampaign.ts:160`

`acceptReflow` updates Supabase + memory but not `writeCachedCampaign`.

Required:

- Update persistent offline cache after accepted reflow.
- Verify cold-start behavior.

**Gemini 3.1 Pro.**

---

### #7 Session/login unhandled rejections

`src/auth/useSession.ts:15`
`src/auth/Login.tsx:27`

Required:

- `.catch`
- mounted/unmounted guard
- loading/working state must recover after failure
- no stale UI state

**Gemini 3.6 Flash.**

---

### #8 Week-label parser

`src/domain/parseTracker.ts:38`

Current `label.split("-")` fails for en/em dashes.

Support:

- `-`
- `–`
- `—`

Use an appropriate delimiter strategy such as `/[–—-]/`.

Add regression tests.

**Gemini 3.6 Flash.**

---

### #9 Content deadline source of truth

`src/domain/planAnalytics.ts:162`

Do not derive the content deadline from week-ending dates.

Use the canonical `CONTENT_DEADLINE` constant.

Add a regression test.

**Gemini 3.6 Flash.**

---

### #10 Stale fallback timer

`src/campaign/useCampaign.ts:98`

Clear the 2.5-second fallback timer during cleanup/unmount.

Add a regression test if the existing testing architecture supports it.

**Gemini 3.6 Flash.**

---

### #11 Undefined patch semantics

`src/adapters/progressRow.ts:29`

Current behavior treats explicit `undefined` as `null`.

Required semantics:

- `undefined` → leave field unchanged
- `null` → explicitly clear field

Use the appropriate condition, e.g. checking `patch[field] !== undefined`.

Add regression coverage.

**Gemini 3.6 Flash.**

---

### #12 E2E clock initialization

`e2e/daily-loop.spec.ts:166`

Set fixed time **before** `page.goto()`.

Required:

- first render must use deterministic test time
- remove wall-clock dependence

**Gemini 3.6 Flash.**

---

# Severity 3 — HIGH PAYOFF

## #13 Eliminate per-keystroke 353-problem recomputation

Files:

- `src/tracker/Tracker.tsx:235`
- `src/campaign/Campaign.tsx:24`

Current behavior:

- Trick input fires `onEdit` on every keystroke.
- Campaign memo depends on the entire loaded object.
- `analysePlan` reruns against ~353 Problems for every character.

Required:

1. Maintain local Trick input state.
2. Commit on blur or debounce.
3. Prevent global plan mutation per keystroke.
4. Narrow memo dependencies to:

`[loaded.problems, loaded.weeks, day]`

5. Preserve existing behavior.
6. Add a regression/performance-oriented test where practical.

**Delegate to Gemini 3.1 Pro.**

This is one of the three highest-priority changes.

---

## #14 Autofocus Trick input

`src/tracker/Tracker.tsx:204`

When marking a problem solved:

- reveal Trick field
- automatically focus the Trick input

This is a high-frequency interaction and should require minimal user movement.

**Gemini 3.6 Flash.**

---

## #15 Make Reflow proposals actionable

`src/plan/Reflow.tsx:136`

Current UI only displays aggregate weekly counts.

Required display:

`PROBLEM_CODE — TITLE — W(from) → W(to)`

Users must be able to understand exactly what they are accepting before committing.

Do not write anything to persistence until acceptance.

**Gemini 3.6 Flash.**

This is one of the three highest-priority changes.

---

## #16 Cap search rendering

`src/tracker/Tracker.tsx:77`

Search currently expands all matching Weeks, potentially rendering 353 unvirtualized rows.

Required:

- cap rendered results per section
- preserve useful search behavior
- avoid unnecessary DOM growth

**Gemini 3.6 Flash.**

---

# Severity 4 — HYGIENE

### #17 Tooling

Add ESLint + Prettier.

**Gemini 3.6 Flash.**

---

### #18 TypeScript safety

`tsconfig.json:7`

Evaluate enabling:

`noUncheckedIndexedAccess`

Do not blindly enable it if it creates excessive unrelated churn.

If enabled:

- fix resulting issues cleanly
- keep scope controlled

**Gemini 3.1 Pro should assess; Flash can implement mechanical fixes.**

---

### #19 Date arithmetic tests

`src/domain/dates.ts`

Add dedicated tests for:

- `daysInclusive`
- `shiftDays`
- DST transitions
- month boundaries
- year boundaries
- Dec 30 extension behavior

**Gemini 3.1 Pro** for test design, then Flash may implement straightforward tests.

---

### #20 Seed rerun guard

`scripts/seed.ts:37`

Current guard only checks problem count.

Required:

- detect partially seeded DB
- avoid proceeding into guaranteed unique-constraint failure
- produce a clear diagnostic

**Gemini 3.6 Flash.**

---

### #21 Copy

`src/tracker/Tracker.tsx:65`

Replace:

`not yet synced`

with:

`not yet saved`

**Gemini 3.6 Flash.**

---

### #22 Mobile hit targets

`src/styles.css:126`

Increase `week-toggle` and `view-switch` hit targets to at least 32px.

Prefer accessible touch targets without unnecessarily changing visual layout.

**Gemini 3.6 Flash.**

---

# Execution Order

Do NOT execute strictly as 1 → 22 if parallelization is safe.

Use these phases.

## Phase 1 — Critical foundations

Run in parallel where safe:

**Gemini 3.1 Pro**

- #1 migration system
- #3 write queue persistence failure
- #4 reflow capacity

**Gemini 3.6 Flash**

- #2 Playwright bootstrap

Wait for these before continuing.

---

# Phase 2 — Related correctness

**Gemini 3.1 Pro**

- #5 queue retry/failure state
- #6 cache update
- #13 performance architecture

These should be reviewed together because #3/#5 share queue behavior and #13 is a major application-performance path.

**Gemini 3.6 Flash**

- #7
- #8
- #9
- #10
- #11
- #12

These are sufficiently isolated.

---

# Phase 3 — UX

**Gemini 3.6 Flash**

- #14
- #15
- #16

---

# Phase 4 — Hygiene

**Gemini 3.6 Flash**

- #17
- #20
- #21
- #22

**Gemini 3.1 Pro**

- assess #18
- design #19

---

# Agent Instructions

For every Gemini task:

1. Inspect only the necessary files first.
2. Understand existing conventions before editing.
3. Make the smallest correct change.
4. Do not perform unrelated refactors.
5. Add regression tests for behavior changes.
6. Run the narrowest relevant tests.
7. Report:
   - files changed
   - behavior changed
   - tests added/modified
   - tests executed
   - failures
   - unresolved concerns

Do not ask the Gemini agent to produce lengthy explanations.

Return concise machine-readable work summaries.

---

# Opus Review Protocol

After each phase, review only the agent summaries and relevant diffs.

Do NOT reread the entire repository unless necessary.

For each change ask:

1. Does it actually fix the reported root cause?
2. Could it introduce data loss?
3. Does it preserve existing behavior?
4. Are tests adequate?
5. Is there an obvious race/lifecycle/state-management problem?
6. Is the implementation unnecessarily complex?

If the implementation is correct, **approve it immediately**.

Do not spend tokens rewriting correct code for stylistic reasons.

---

# Critical Acceptance Criteria

Before declaring completion:

### Database

- Running migration twice produces no destructive changes.
- Existing lesson titles survive repeated migration execution.
- Migration tracking works for fresh and existing databases.

### E2E

- Fresh clone can bootstrap authentication.
- Playwright config does not crash before login setup.

### Queue

- localStorage failure cannot crash editing.
- Queue remains recoverable in memory.
- Failed writes retry with bounded backoff.
- Permanent failure becomes visible.

### Reflow

- Never schedules beyond target ceiling.
- Regression test proves the overflow case is fixed.

### Performance

- Typing a Trick does NOT invoke full `analysePlan` on every keystroke.
- Memoization dependencies are appropriately narrow.

### Cache

- Accepted reflow updates server, memory, and offline cache consistently.

### Async lifecycle

- No stale timers.
- No unhandled auth rejections.
- Unmounted components cannot update stale state.

### UX

- Solved → Trick input receives focus.
- Reflow shows exact moves.
- Search does not render hundreds of unnecessary rows.
- Mobile controls meet the minimum touch-target requirement.

### Tests

Run the full test suite after integration.

Also run:

- typecheck
- lint
- relevant E2E tests
- migration tests
- domain/reflow tests

---

# Final Opus Report

At the end, provide ONLY:

## Status

PASS / PASS WITH CONCERNS / BLOCKED

## Fixed

- #...

## Remaining

- #...

## Validation

- Tests: ...
- Typecheck: ...
- Lint: ...
- E2E: ...

## Risks

- ...

Do not provide a long narrative.

The objective is **correct code with minimum Opus token consumption**, not maximum explanation.
