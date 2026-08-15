"""Run identity: the manifest that makes a number traceable, and the audit that checks it.

Promoted from `notebooks/08_6_reproducible_eval_harness.ipynb`, whose config hashing is already the
right primitive (`config.canonical_hash`) — the notebook's in-memory record list is what changed, to
streamed `records.jsonl`, since 2M-corpus runs do not fit its shape.

**A run is a prefix, not a directory.** The notebook's plan was `runs/<run_id>/`, and this module's
own docstring said so until the layout was reconciled with the repo: `runs/` is gitignored, and every
artifact any number rests on lives under `docs/harvest/<name>.*` — which is where the G0 records had
to be moved in W1 for exactly this reason. So one run is:

    <prefix>.manifest.json   config hash, index fingerprint, split hash, git sha, model ids, times
    <prefix>.records.jsonl   one QueryRecord per (question, system, seed)
    <prefix>.costs.jsonl     one CostRecord per billable or timed unit of work

**G5 is the reason this exists**: every cell of Tables 1–5 must be populated from a run manifest,
with CIs. A number whose manifest cannot be produced does not go in the paper.

Four rules, each one a failure this module refuses or reports rather than repairs:

1. **The manifest is written before the run, not after.** `run_manifest()` stamps `started_at` and
   the four identities; `finalize_run()` stamps `finished_at` and the row counts. Assembled at the
   end, the manifest would record the tree as it looked once the run finished, and a crashed run
   would be indistinguishable from a complete one. That is `corpus_manifest.json`'s `n_prescan_rows`
   lesson: a count is evidence only when it is recorded before the thing that can invalidate it.
2. **A prefix holds one run.** Writing a second manifest over the first hands new records the old
   run's provenance, and nothing downstream can see it — the records are fresh, the manifest is
   well-formed, and the config hash belongs to something else.
3. **A manifest written after the fact says so.** `backfill_manifest()` is the only way to manifest
   an artifact that was produced before this module existed. It cannot invent what the run did not
   record: its git sha is the commit that *added the records file*, its `unrecovered` list names the
   fields nobody can recover, and `verify_run()` reports it as a caveat forever.
4. **`verify_run()` reports, never repairs.** Same contract as `QueryRecord.validate()`: an empty
   list means the prefix is quotable without a caveat. It is what a table-building script calls over
   every prefix it is about to read, so a missing manifest is a finding, not an exception.

The manifest carries the whole config, not only its hash: a hash cannot be inverted, and a table
caption names the knobs that differ (`config.config_diff`), which needs the values. It carries no
number derived from the records — those are recomputed from `records.jsonl` by `scoring/`, which is
what keeps a re-score from being a re-run.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import RunConfig
from .data import load_splits
from .schema import SCHEMA_VERSION, read_jsonl

MANIFEST_SUFFIX = ".manifest.json"
RECORDS_SUFFIX = ".records.jsonl"
COSTS_SUFFIX = ".costs.jsonl"

#: 1.0.0 is the shape below. Bump when a field is added, so a manifest cannot claim a shape it does
#: not have — the same reason `QueryRecord.schema_version` is stored rather than assumed.
MANIFEST_VERSION = "1.0.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def manifest_path(prefix: Path) -> Path:
    return Path(str(prefix) + MANIFEST_SUFFIX)


def records_path(prefix: Path) -> Path:
    return Path(str(prefix) + RECORDS_SUFFIX)


def costs_path(prefix: Path) -> Path:
    return Path(str(prefix) + COSTS_SUFFIX)


def git_sha(repo: Path | None = None) -> str:
    """`<40-hex>`, suffixed `-dirty` when the tree has uncommitted changes, or `"unknown"`.

    The suffix is the point. A result is meant to be reproducible from a commit, and a run made from
    a dirty tree is reproducible from nothing, so recording the bare sha would assert the opposite.
    `"unknown"` is returned rather than raised for the released-tarball case, where there is no
    repository and refusing to run would be the wrong trade.
    """
    root = Path(repo) if repo else _REPO_ROOT
    try:
        sha = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def commit_that_added(path: Path) -> str:
    """The oldest commit touching `path`, or `"unknown"`.

    This is the only sha a backfilled manifest can honestly carry: the run's own tree state was
    never recorded, and the commit that first brought the records into the repository is an upper
    bound on it that is checkable by anyone. It is **not** the same claim as `git_sha()` and is
    labelled as such in the manifest.
    """
    path = Path(path)
    try:
        log = _git(path.parent, "log", "--reverse", "--format=%H", "--", str(path.resolve()))
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return log.splitlines()[0] if log else "unknown"


def run_manifest(
    config: RunConfig,
    prefix: Path,
    *,
    run_id: str | None = None,
    arms: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """Write `<prefix>.manifest.json`. Call this **before** the first record.

    `run_id` defaults to the prefix's name, which is what ties the manifest to the records beside it.
    Pass it only when the records already carry a different one — a run cannot be renamed after the
    fact, and `verify_run()` reports the disagreement if the two drift.

    `arms` is for an artifact that is legitimately several runs in one file: Table 1's four ablation
    rows share an index, a split and a pool, differ by `{bm25, dense, rrf, rerank}`, and land in one
    `records.jsonl` under one `run_id` each. `{arm_run_id: {dotted knob: value}}` gives every arm its
    own `config_hash` — which is what G5 needs, since each arm is a different *cell* — while the
    top-level config stays the shared base. Without it those rows would read as foreign records.
    """
    return _write_manifest(
        _fields(
            config,
            run_id=run_id if run_id is not None else Path(prefix).name,
            git_sha_value=git_sha(),
            provenance={"kind": "live"},
            arms=arms,
        ),
        prefix,
    )


def backfill_manifest(
    config: RunConfig,
    prefix: Path,
    *,
    run_id: str,
    started_at: str | None,
    finished_at: str | None,
    source: str,
    unrecovered: Sequence[str],
    arms: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict:
    """Manifest an artifact produced before this module existed. Written already finalized.

    Every argument exists because the value cannot be derived here and must come from the artifact
    itself: `run_id` from its records, the timestamps from what the run wrote down, `source` naming
    the file the knobs were read out of, and `unrecovered` naming the knobs that file does not carry.
    `unrecovered` is not a formality — a backfilled config hash is a hash of partly *default* values,
    so it does not identify the run, and `verify_run()` says so on every read.
    """
    if not unrecovered:
        raise ValueError(
            "unrecovered is empty, which claims the artifact recorded every knob. If that is true "
            "it does not need backfilling — write the manifest with run_manifest() at run time."
        )
    manifest = _fields(
        config,
        run_id=run_id,
        git_sha_value=commit_that_added(records_path(prefix)),
        provenance={
            "kind": "backfilled",
            "source": source,
            "unrecovered": list(unrecovered),
            "git_sha_is": "the commit that added the records, not the tree the run was made from",
            "backfilled_at": _now(),
        },
        arms=arms,
    )
    manifest["started_at"] = started_at
    manifest["finished_at"] = finished_at
    manifest["n_records"] = _count_lines(records_path(prefix))
    manifest["n_costs"] = _count_lines(costs_path(prefix))
    return _write_manifest(manifest, prefix)


def load_manifest(prefix: Path) -> dict:
    return json.loads(manifest_path(prefix).read_text(encoding="utf-8"))


def finalize_run(prefix: Path) -> dict:
    """Stamp `finished_at` and count the rows that are on disk **now**.

    The counts are not carried from the loop that wrote them: what a later reader can check is the
    file, so the number recorded has to be the file's. It is the only guard that can see a
    truncation, since `write_jsonl` opens with `"w"`.
    """
    if not manifest_path(prefix).exists():
        raise FileNotFoundError(
            f"{prefix} has no manifest, so this run cannot be finalized. A manifest written now "
            "would record the tree and the clock at finalize time, not at run time. Call "
            "run_manifest() before the run — or backfill_manifest(), which says that it did."
        )
    manifest = load_manifest(prefix)
    manifest["n_records"] = _count_lines(records_path(prefix))
    manifest["n_costs"] = _count_lines(costs_path(prefix))
    manifest["finished_at"] = _now()
    return _write_manifest(manifest, prefix, replacing=True)


def verify_run(prefix: Path) -> list[str]:
    """Report every reason this prefix is not quotable as it stands. Empty list means it is.

    Reports, never repairs — a violation is a measurement about the run, and the caller is a table
    script auditing many prefixes, so one bad prefix must not stop the audit.
    """
    if not manifest_path(prefix).exists():
        return [f"{manifest_path(prefix).name}: missing"]

    try:
        manifest = load_manifest(prefix)
    except json.JSONDecodeError as exc:
        return [f"{manifest_path(prefix).name}: not readable as JSON ({exc})"]

    problems: list[str] = []
    if manifest.get("finished_at") is None:
        problems.append(
            "manifest: finished_at is null — the run never reached finalize_run(), so the records "
            "may be a prefix of the run this config describes"
        )
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        problems.append(
            f"manifest: manifest_version {manifest.get('manifest_version')!r}, this code writes "
            f"{MANIFEST_VERSION!r}"
        )
    provenance = manifest.get("provenance") or {}
    if provenance.get("kind") != "live":
        problems.append(
            f"manifest: provenance is {provenance.get('kind')!r} from "
            f"{provenance.get('source')!r} — {', '.join(provenance.get('unrecovered', [])) or 'no'} "
            "knobs are defaults rather than what ran, and git_sha is "
            f"{provenance.get('git_sha_is', 'not the run tree')}"
        )
    if str(manifest.get("git_sha", "")).endswith("-dirty"):
        problems.append(
            "manifest: git_sha is dirty — the run was made from uncommitted code, so it is not "
            "reproducible from any commit"
        )

    arms = manifest.get("arms") or {}
    declared = {manifest.get("run_id"), *arms}
    for path, recorded in ((records_path(prefix), "n_records"), (costs_path(prefix), "n_costs")):
        n = _count_lines(path)
        expected = manifest.get(recorded)
        if expected is not None and n != expected:
            problems.append(
                f"{path.name}: holds {n} rows, the manifest records {expected} — the file changed "
                "after the run was finalized"
            )
        present = _distinct(path, "run_id")
        foreign = sorted(present - declared)
        if foreign:
            problems.append(
                f"{path.name}: carries rows from {', '.join(repr(r) for r in foreign)}, which the "
                f"manifest declares neither as its run_id ({manifest.get('run_id')!r}) nor as an arm"
            )
        missing = sorted(set(arms) - present) if path.exists() and arms else []
        if missing:
            problems.append(
                f"{path.name}: declares arms {', '.join(repr(a) for a in missing)} that have no "
                "rows — a table cell with a config hash and no records behind it"
            )

    stale = sorted(_distinct(records_path(prefix), "schema_version") - {manifest.get("schema_version")})
    if stale:
        problems.append(
            f"{records_path(prefix).name}: schema_version {', '.join(repr(s) for s in stale)} "
            f"against the manifest's {manifest.get('schema_version')!r} — a field a table needs may "
            "be absent with no parse error anywhere"
        )
    return problems


# ---------------------------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------------------------


def _fields(
    config: RunConfig,
    *,
    run_id: str,
    git_sha_value: str,
    provenance: dict[str, Any],
    arms: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "manifest_version": MANIFEST_VERSION,
        "config_hash": config.hash(),
        "config_version": config.config_version,
        "schema_version": SCHEMA_VERSION,
        "index_fingerprint": config.index_fingerprint(),
        "split": config.split,
        "split_hash": load_splits()["hash"],
        "git_sha": git_sha_value,
        "seeds": list(config.generation.seeds),
        "models": {
            "generator": config.generation.model,
            "dense_encoder": config.retrieval.dense_encoder,
            "query_encoder": config.retrieval.query_encoder,
            "reranker": config.retrieval.reranker,
            "verifier": config.verifier.model,
            "judge": config.verifier.judge_model,
        },
        "started_at": _now(),
        "finished_at": None,
        "n_records": None,
        "n_costs": None,
        "arms": {
            # `ablate` is the same primitive the sweep itself uses, so an arm's hash here is the hash
            # the arm would have produced had it been run alone under its own prefix.
            arm_id: {"config_hash": config.ablate(arm_id, **dict(diff)).hash(), "differs": dict(diff)}
            for arm_id, diff in (arms or {}).items()
        },
        "provenance": provenance,
        "config": asdict(config),
    }


def _write_manifest(manifest: dict[str, Any], prefix: Path, *, replacing: bool = False) -> dict:
    path = manifest_path(prefix)
    if path.exists() and not replacing:
        prior = json.loads(path.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"{path} already holds a manifest ({prior.get('config_hash')}, started "
            f"{prior.get('started_at')}). Writing over it would give these records that run's "
            "provenance. Use a new prefix, or delete the old run's files deliberately."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Read back rather than returning the dict just built: JSON has no tuple, and
    # `GenerationConfig.stop` / `seeds` are tuples, so the in-memory shape is not the shape on disk.
    # The file is the artifact of record, so it is what callers get.
    return load_manifest(prefix)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _distinct(path: Path, field: str) -> set[str]:
    """Stringified so a row missing the field sorts and reports instead of raising in `sorted`."""
    if not path.exists():
        return set()
    return {str(row.get(field)) for row in read_jsonl(path)}


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
