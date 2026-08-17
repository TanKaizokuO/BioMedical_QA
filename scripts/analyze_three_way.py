#!/usr/bin/env python3
import json
import statistics
from pathlib import Path
from biomedqa.schema import read_query_records, read_jsonl, CostRecord, System
from biomedqa.scoring.granularity import arm_granularity
from biomedqa.scoring.citation import citation_f1

def load_run(prefix_str):
    prefix = Path(prefix_str)
    rec_path = prefix.with_name(f"{prefix.name}.records.jsonl")
    cost_path = prefix.with_name(f"{prefix.name}.costs.jsonl")
    sum_path = prefix.with_name(f"{prefix.name}.summary.json")

    records = list(read_query_records(rec_path))
    costs = [CostRecord(**d) for d in read_jsonl(cost_path)]
    summary = json.loads(sum_path.read_text())

    return records, costs, summary

def analyze_all():
    runs = {
        "baseline": load_run("docs/harvest/generate_fp05_n100_baseline"),
        "guided_unbatched": load_run("docs/harvest/generate_fp05_n100_guided"),
        "guided_batched": load_run("docs/harvest/generate_fp05_n100_guided_batched"),
    }

    results = {}

    for run_name, (recs, costs, summary) in runs.items():
        ph_recs = [r for r in recs if r.system == System.POST_HOC]
        ph_summary = summary["per_system"]["post_hoc"]

        # Clean parses (query records with 0 errors)
        clean_parses = ph_summary["clean_parses"]
        total_queries = len(ph_recs)
        clean_parse_rate = clean_parses / total_queries if total_queries else 0

        # Claims & valid claim parse rate
        # Each record has r.claims. If a record has parse errors (truncation / JSON error),
        # how many claims were validly parsed vs lost?
        # r.errors holds errors for that record.
        # Let's count claims in clean records vs total claims across all records.
        total_claims = sum(len(r.claims) for r in ph_recs)
        valid_parse_claims = sum(len(r.claims) for r in ph_recs if not summary_row_errors(summary, r.query_id, "post_hoc"))
        valid_claim_parse_rate = valid_parse_claims / total_claims if total_claims else 0

        # Zero-citation claims
        zero_cit_claims = 0
        total_citations = 0
        for r in ph_recs:
            for c in r.claims:
                cits = len(c.citations)
                total_citations += cits
                if cits == 0:
                    zero_cit_claims += 1

        zero_cit_pct = (zero_cit_claims / total_claims) * 100 if total_claims else 0

        # quote_not_found
        qnf = ph_summary.get("quote_not_found", 0)

        # Call failures
        call_failures = ph_summary.get("call_failure_count", 0)

        # Truncations / JSON failures
        truncations = sum(1 for r in ph_recs if summary_row_errors(summary, r.query_id, "post_hoc"))

        # Stage-2 batch call distribution and input tokens
        # Group costs for post-hoc stage-2 calls
        # In post_hoc, stage 1 is component="generate" (or first cost), stage 2+ are cite calls
        stg2_costs = []
        for c in costs:
            # post_hoc costs have query_id ending or matching, or we can check component
            # let's inspect cost records for post_hoc
            pass

        results[run_name] = {
            "n_queries": total_queries,
            "clean_parses": clean_parses,
            "clean_parse_rate": clean_parse_rate,
            "total_claims": total_claims,
            "valid_parse_claims": valid_parse_claims,
            "valid_claim_parse_rate": valid_claim_parse_rate,
            "claims_per_query": total_claims / total_queries if total_queries else 0,
            "total_citations": total_citations,
            "zero_cit_claims": zero_cit_claims,
            "zero_cit_pct": zero_cit_pct,
            "quote_not_found": qnf,
            "call_failures": call_failures,
            "truncations": truncations,
        }

    return results

def summary_row_errors(summary, query_id, system):
    for row in summary.get("rows", []):
        if row.get("query_id") == query_id and row.get("system") == system:
            return row.get("errors", [])
    return []

if __name__ == "__main__":
    res = analyze_all()
    print(json.dumps(res, indent=2))
