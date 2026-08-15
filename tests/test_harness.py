"""Run identity: `<prefix>.manifest.json` beside `<prefix>.records.jsonl` and `.costs.jsonl`.

G5 refuses any table cell whose manifest cannot be produced, so every failure below has the same
shape — **a number that looks traceable and is not**:

- `test_a_second_run_cannot_inherit_the_first_manifest` — writing a manifest over an existing one
  hands the new records the old run's config hash, index fingerprint and git sha. Nothing downstream
  can see it: the records are fresh and the manifest is well-formed.
- `test_an_unfinished_run_is_reported_as_unfinished` — `finished_at` is stamped after the records are
  on disk, so a crashed run leaves it null. That manifest must not read as a completed run whose
  numbers are quotable.
- `test_records_from_another_run_are_reported` — `run_id` ties a record to the manifest beside it. A
  `records.jsonl` assembled from two runs scores as one, under one run's config hash.
- `test_a_truncated_records_file_is_reported` — the counts are written at finalize time from the file
  as it stood, and `write_jsonl` opens with `"w"`. A later truncation is invisible to every other
  guard, exactly as `corpus_manifest.json`'s `n_prescan_rows` is the only thing that can see a
  truncated prescan.
- `TestBackfill` — a manifest written after the fact for an artifact that predates this module. The
  hazard is that it looks identical to a live one while its config is partly defaults and its git sha
  is not the tree the run was made from. Both facts have to survive into every read.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from biomedqa.config import RunConfig, config_diff
from biomedqa.data import load_splits
from biomedqa.harness import (
    MANIFEST_VERSION,
    backfill_manifest,
    costs_path,
    finalize_run,
    git_sha,
    load_manifest,
    manifest_path,
    records_path,
    run_manifest,
    verify_run,
)
from biomedqa.schema import CostRecord, QueryRecord, System, read_jsonl, to_dict, write_jsonl

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def _config(name: str = "run-1") -> RunConfig:
    return RunConfig().ablate(name, **{"generation.model": MODEL})


def _write_run(prefix, run_ids=("run-1",), n_records: int = 2, n_costs: int = 2) -> None:
    write_jsonl(
        records_path(prefix),
        [
            QueryRecord(run_id=r, query_id=str(i), question="q?", system=System.JOINT, seed=0)
            for r in run_ids
            for i in range(n_records)
        ],
    )
    write_jsonl(
        costs_path(prefix),
        [
            CostRecord(run_id=r, query_id=str(i), component="generate", backend="vllm:x")
            for r in run_ids
            for i in range(n_costs)
        ],
    )


def _pin_sha(prefix, sha: str) -> None:
    manifest = load_manifest(prefix)
    manifest["git_sha"] = sha
    manifest_path(prefix).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


class TestManifest:
    def test_manifest_records_every_identity_a_number_is_traced_by(self, tmp_path):
        """Four hashes and a commit, each of which can change while the others do not: the knobs
        (`config_hash`), the index the passages came from (`index_fingerprint`), the questions
        (`split_hash`), and the code (`git_sha`)."""
        config = _config()
        prefix = tmp_path / "run-1"
        manifest = run_manifest(config, prefix)

        assert manifest_path(prefix).name == "run-1.manifest.json"
        assert load_manifest(prefix) == manifest
        assert manifest["run_id"] == "run-1"
        assert manifest["manifest_version"] == MANIFEST_VERSION
        assert manifest["config_hash"] == config.hash()
        assert manifest["index_fingerprint"] == config.index_fingerprint()
        assert manifest["split_hash"] == load_splits()["hash"]
        assert manifest["git_sha"] == git_sha()
        assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?|unknown", manifest["git_sha"])
        assert manifest["provenance"] == {"kind": "live"}
        assert manifest["started_at"].endswith("+00:00")
        assert manifest["finished_at"] is None

    def test_the_knobs_are_recoverable_from_the_manifest_alone(self, tmp_path):
        """A hash cannot be inverted, and a table caption names the knobs that differ
        (`config_diff`) rather than reprinting them. So the manifest carries the config itself —
        otherwise the caption can only be written by the process still holding the `RunConfig`."""
        config = _config()
        manifest = run_manifest(config, tmp_path / "run-1")

        assert manifest["config"]["generation"]["model"] == MODEL
        assert manifest["config"]["name"] == "run-1"
        assert config_diff(RunConfig(), config) == {
            "name": ("base", "run-1"),
            "generation.model": ("", MODEL),
        }

    def test_the_model_ids_are_named_where_a_reader_will_look(self, tmp_path):
        """Tables 3 and 4 compare models, not configs. Three levels down in `config` is enough for a
        machine and not enough for the person checking which verifier produced a row."""
        manifest = run_manifest(_config(), tmp_path / "run-1")
        base = RunConfig()

        assert manifest["models"] == {
            "generator": MODEL,
            "dense_encoder": base.retrieval.dense_encoder,
            "query_encoder": base.retrieval.query_encoder,
            "reranker": base.retrieval.reranker,
            "verifier": base.verifier.model,
            "judge": base.verifier.judge_model,
        }

    def test_a_second_run_cannot_inherit_the_first_manifest(self, tmp_path):
        """The overwrite is the dangerous direction and refusing is the cheap one: a re-run takes a
        new prefix, and what happens to the old files is the operator's decision."""
        run_manifest(_config(), tmp_path / "run-1")
        with pytest.raises(FileExistsError, match="already holds a manifest"):
            run_manifest(_config("other"), tmp_path / "run-1")

    def test_the_run_id_may_be_given_when_the_records_already_carry_one(self, tmp_path):
        """`parity_iter0` through `parity_iter1b` all stamp `run_id: w4_generate_smoke`, so for those
        artifacts the prefix name is not the run id. Renaming the run to match the prefix would edit
        the records; naming it here does not."""
        manifest = run_manifest(_config(), tmp_path / "parity_iter9", run_id="w4_generate_smoke")

        assert manifest["run_id"] == "w4_generate_smoke"


class TestFinalize:
    def test_finalize_stamps_the_end_and_counts_what_is_on_disk(self, tmp_path):
        prefix = tmp_path / "run-1"
        run_manifest(_config(), prefix)
        _write_run(prefix, n_records=3, n_costs=5)

        manifest = finalize_run(prefix)

        assert manifest["n_records"] == 3
        assert manifest["n_costs"] == 5
        assert manifest["finished_at"] is not None
        assert load_manifest(prefix) == manifest

    def test_finalize_refuses_a_prefix_with_no_manifest(self, tmp_path):
        """A run whose manifest was never written cannot be given one at the end: every identity in
        it — the git sha included — would describe finalize time, not run time."""
        prefix = tmp_path / "run-1"
        _write_run(prefix)
        with pytest.raises(FileNotFoundError, match="no manifest"):
            finalize_run(prefix)


class TestVerify:
    def test_a_finished_run_verifies_clean(self, tmp_path):
        """`git_sha` is pinned to a committed-looking sha, because the repo this suite runs in is
        usually dirty and the point here is the rest of the audit."""
        prefix = tmp_path / "run-1"
        run_manifest(_config(), prefix)
        _write_run(prefix)
        finalize_run(prefix)
        _pin_sha(prefix, "a" * 40)

        assert verify_run(prefix) == []

    def test_a_run_made_from_a_dirty_tree_is_reported(self, tmp_path):
        """The one identity that cannot be recovered later. A number produced from uncommitted code
        is reproducible from nothing, and `git_sha` recording the bare sha would say the opposite."""
        prefix = tmp_path / "run-1"
        run_manifest(_config(), prefix)
        _write_run(prefix)
        finalize_run(prefix)
        _pin_sha(prefix, "a" * 40 + "-dirty")

        assert any("dirty" in p for p in verify_run(prefix))

    def test_an_unfinished_run_is_reported_as_unfinished(self, tmp_path):
        prefix = tmp_path / "run-1"
        run_manifest(_config(), prefix)
        _write_run(prefix)

        assert any("finished_at" in p for p in verify_run(prefix))

    def test_records_from_another_run_are_reported(self, tmp_path):
        prefix = tmp_path / "run-1"
        run_manifest(_config(), prefix)
        _write_run(prefix, run_ids=("run-1", "run-0"))
        finalize_run(prefix)

        problems = verify_run(prefix)

        assert any("records.jsonl" in p and "run-0" in p for p in problems)
        assert any("costs.jsonl" in p and "run-0" in p for p in problems)

    def test_a_truncated_records_file_is_reported(self, tmp_path):
        prefix = tmp_path / "run-1"
        run_manifest(_config(), prefix)
        _write_run(prefix, n_records=3)
        finalize_run(prefix)
        _write_run(prefix, n_records=2)  # `write_jsonl` opens with "w"

        assert any(
            "records.jsonl" in p and "2 rows" in p and "3" in p for p in verify_run(prefix)
        )

    def test_a_schema_version_the_records_do_not_carry_is_reported(self, tmp_path):
        """The manifest names the schema its records were written under. An older record is not a
        parse error — `query_record_from_dict` reads it — so nothing else notices that a field the
        table needs may simply be absent."""
        prefix = tmp_path / "run-1"
        run_manifest(_config(), prefix)
        row = to_dict(
            QueryRecord(run_id="run-1", query_id="1", question="q?", system=System.JOINT, seed=0)
        )
        row["schema_version"] = "0.9.0"
        records_path(prefix).write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        finalize_run(prefix)

        assert any("schema_version" in p and "0.9.0" in p for p in verify_run(prefix))

    def test_a_missing_manifest_is_a_reported_problem_not_a_crash(self, tmp_path):
        """`verify_run` is what a table script calls over every prefix it is about to read. Raising
        on the un-manifested one would stop the audit at the first finding instead of listing it."""
        assert verify_run(tmp_path / "run-1") == ["run-1.manifest.json: missing"]


class TestArms:
    """Table 1 is four ablation rows in one `records.jsonl`, one `run_id` each. Each row is a
    different table *cell*, so each needs its own config hash — and without `arms` the other three
    rows read as records from a foreign run."""

    ARMS = {
        "table1_bm25_only": {"retrieval.bm25": True, "retrieval.dense": False,
                             "retrieval.rrf": False, "retrieval.rerank": False},
        "table1_dense_only": {"retrieval.bm25": False, "retrieval.dense": True,
                              "retrieval.rrf": False, "retrieval.rerank": False},
    }

    def test_each_arm_carries_the_hash_it_would_have_had_on_its_own(self, tmp_path):
        config = _config("table1")
        manifest = run_manifest(config, tmp_path / "table1", arms=self.ARMS)

        for arm_id, differs in self.ARMS.items():
            assert manifest["arms"][arm_id]["differs"] == differs
            assert manifest["arms"][arm_id]["config_hash"] == config.ablate(arm_id, **differs).hash()
        bm25, dense = (manifest["arms"][a]["config_hash"] for a in self.ARMS)
        assert bm25 != dense, "two arms differing in retrieval flags must not share a hash"

    def test_arm_records_are_not_foreign_records(self, tmp_path):
        prefix = tmp_path / "table1"
        run_manifest(_config("table1"), prefix, arms=self.ARMS)
        _write_run(prefix, run_ids=tuple(self.ARMS))
        finalize_run(prefix)
        _pin_sha(prefix, "a" * 40)

        assert verify_run(prefix) == []

    def test_an_arm_with_no_records_is_reported(self, tmp_path):
        """A declared arm with no rows is a table cell with a config hash and nothing behind it —
        the exact shape of a row that silently did not run."""
        prefix = tmp_path / "table1"
        run_manifest(_config("table1"), prefix, arms=self.ARMS)
        _write_run(prefix, run_ids=("table1_bm25_only",))
        finalize_run(prefix)

        assert any("table1_dense_only" in p and "have no" in p for p in verify_run(prefix))


class TestBackfill:
    def _backfill(self, prefix, **over):
        kwargs = dict(
            run_id="w4_generate_smoke",
            started_at="2026-08-13T19:00:00+00:00",
            finished_at="2026-08-13T19:13:08.658354+00:00",
            source="parity_iter9.summary.json",
            unrecovered=["retrieval.*", "generation.seeds"],
        )
        kwargs.update(over)
        return backfill_manifest(_config("parity_iter9"), prefix, **kwargs)

    def test_a_backfilled_manifest_keeps_the_times_the_run_recorded(self, tmp_path):
        """`_now()` would date the artifact to the backfill, which is the one thing a reader would
        use it for."""
        prefix = tmp_path / "parity_iter9"
        _write_run(prefix, run_ids=("w4_generate_smoke",), n_records=3, n_costs=4)

        manifest = self._backfill(prefix)

        assert manifest["started_at"] == "2026-08-13T19:00:00+00:00"
        assert manifest["finished_at"] == "2026-08-13T19:13:08.658354+00:00"
        assert manifest["n_records"] == 3
        assert manifest["n_costs"] == 4

    def test_a_backfilled_run_never_verifies_clean(self, tmp_path):
        """The caveat is permanent and has to reach whoever quotes the number: a backfilled config
        hash is a hash over partly default values, so it does not identify the run."""
        prefix = tmp_path / "parity_iter9"
        _write_run(prefix, run_ids=("w4_generate_smoke",))
        self._backfill(prefix)

        problems = verify_run(prefix)

        assert any("backfilled" in p and "retrieval.*" in p for p in problems)
        assert not any("finished_at" in p for p in problems)

    def test_backfill_refuses_to_claim_the_artifact_recorded_everything(self, tmp_path):
        """An empty `unrecovered` is the shape of a backfilled manifest that reads as authoritative.
        If nothing is missing, the run could have written its own manifest."""
        prefix = tmp_path / "parity_iter9"
        _write_run(prefix, run_ids=("w4_generate_smoke",))
        with pytest.raises(ValueError, match="unrecovered is empty"):
            self._backfill(prefix, unrecovered=[])

    def test_the_backfilled_git_sha_is_labelled_as_not_the_run_tree(self, tmp_path):
        """`commit_that_added` is an upper bound on the code that ran, not the code that ran. A bare
        sha in this field would assert reproducibility the artifact cannot support."""
        prefix = tmp_path / "parity_iter9"
        _write_run(prefix, run_ids=("w4_generate_smoke",))

        manifest = self._backfill(prefix)

        assert "not the tree" in manifest["provenance"]["git_sha_is"]
        assert manifest["provenance"]["source"] == "parity_iter9.summary.json"



def test_table1_arm_ids_match_the_ids_its_records_carry():
    """`table1_baseline.arm_run_id` derives an arm's `run_id` from its flags, and
    `backfill_manifests.py` rebuilds the same id from the flags the artifact recorded. If the two ever
    disagree with the committed records, Table 1's manifest declares four arms that no row belongs to
    and every row reads as a foreign record — which is silent until someone audits the table.
    """
    from dataclasses import replace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from table1_baseline import ROWS, _BASE_CONFIG, arm_run_id

    records = Path(__file__).resolve().parents[1] / "docs/harvest/table1_rows_1_4.records.jsonl"
    if not records.exists():
        pytest.skip("table1_rows_1_4 artifacts not present")

    # `read_jsonl`, not `read_text().splitlines()`: the records carry U+2028, which `str.splitlines`
    # treats as a line break and JSON does not (`tests/test_schema_roundtrip.py` pins the hazard).
    on_disk = {row["run_id"] for row in read_jsonl(records)}
    derived = {
        arm_run_id(replace(_BASE_CONFIG, **row["config_overrides"])) for row in ROWS
    }

    assert derived == on_disk