# BioMedical_QA

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues (`TanKaizokuO/BioMedical_QA`), managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Status & Goals Tracking

Agents MUST update `Upcoming_goals.md` as targets are completed or modified. Use ASD-STE100 Simplified Technical English and ubiquitous domain terms from `CONTEXT.md`.

## Working conventions

### Pushing

**Push to `origin/main` without asking.** Standing authorization, granted 2026-08-10; it does not
expire at a session boundary. Earlier handoffs say to ask before every push — that rule is
superseded, and a handoff repeating it should be corrected against this file rather than obeyed.

Always `git pull --rebase` first: the A4000 box pushes directly to `origin/main`.

### Commands for the A4000

The box is copy-paste only — no SSH from the agent environment. Hand over **one-line** commands and
say which machine each belongs to. Multi-line blocks with `\` continuations have silently lost
their flags three times, wasting GPU runs. Every command uses `uv run python`, never bare `python`.
