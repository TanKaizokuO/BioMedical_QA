#!/usr/bin/env python3
import json
import statistics
import math
import time
from pathlib import Path
from biomedqa.schema import read_query_records, read_jsonl, CostRecord, System
from biomedqa.scoring.granularity import arm_granularity
from biomedqa.scoring.citation import citation_f1
from biomedqa.verify import MiniCheckVerifier, phi_from_scores

def load_run(prefix_str):
    prefix = Path(prefix_str)
    rec_path = prefix.with_name(f"{prefix.name}.records.jsonl")
    cost_path = prefix.with_name(f"{prefix.name}.costs.jsonl")
    sum_path = prefix.with_name(f"{prefix.name}.summary.json")

    records = list(read_query_records(rec_path))
    costs = [CostRecord(**d) for d in read_jsonl(cost_path)]
    summary = json.loads(sum_path.read_text())

    return records, costs, summary

def analyze_stage2(records, costs):
    by_q = {}
    for c in costs:
        if c.query_id:
            by_q.setdefault(c.query_id, []).append(c)

    stage2_calls = []
    claims_per_batch = []
    
    for r in records:
        if r.system == System.POST_HOC:
            q_costs = by_q.get(r.query_id, [])
            if len(q_costs) >= 4:
                s2_for_q = q_costs[2:-1]
            elif len(q_costs) == 3:
                s2_for_q = [q_costs[2]]
            else:
                s2_for_q = []
            
            stage2_calls.extend(s2_for_q)
            
            n_claims = len(r.claims)
            n_batches = len(s2_for_q)
            if n_batches > 0:
                rem = n_claims
                for b_idx in range(n_batches):
                    chunk = min(rem, 5) if n_batches > 1 else rem
                    claims_per_batch.append(chunk)
                    rem -= chunk
            elif n_claims > 0:
                claims_per_batch.append(n_claims)

    s2_prompts = [c.input_tokens for c in stage2_calls if c.input_tokens is not None]
    
    def stats(vals):
        if not vals:
            return {"total": 0, "count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "p25": 0, "p75": 0, "p90": 0}
        s = sorted(vals)
        n = len(s)
        return {
            "total": sum(s),
            "count": n,
            "mean": round(statistics.fmean(s), 2),
            "median": round(statistics.median(s), 2),
            "min": s[0],
            "max": s[-1],
            "p25": s[int(n*0.25)],
            "p75": s[int(n*0.75)],
            "p90": s[int(n*0.90)],
        }

    return {
        "stage2_call_count": len(stage2_calls),
        "prompt_tokens": stats(s2_prompts),
        "claims_per_batch": stats(claims_per_batch),
    }

def get_cached_scores(pairs):
    cache_file = Path("docs/harvest/minicheck_cache.json")
    cache = {}
    if cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text())
            cache = {tuple(k.split("|||")): v for k, v in raw.items()}
        except Exception:
            cache = {}

    missing = [p for p in pairs if p not in cache]
    print(f"MiniCheck cache: {len(cache)} cached, {len(missing)} missing out of {len(pairs)} pairs", flush=True)

    if missing:
        import torch
        torch.set_num_threads(16)
        verifier = MiniCheckVerifier(batch_size=64)
        
        # Batch missing into chunks of 100 pairs to save progress incrementally
        chunk_size = 100
        for idx in range(0, len(missing), chunk_size):
            chunk = missing[idx : idx + chunk_size]
            t0 = time.time()
            res = verifier.score_pairs(chunk)
            t1 = time.time()
            for pair, vscore in zip(chunk, res):
                cache[pair] = vscore.score
            
            # Save cache to disk
            serializable = {f"{p[0]}|||{p[1]}": score for p, score in cache.items()}
            cache_file.write_text(json.dumps(serializable))
            print(f"Scored chunk {idx//chunk_size + 1}/{(len(missing)+chunk_size-1)//chunk_size} ({len(chunk)} pairs) in {t1-t0:.1f}s — total cached: {len(cache)}", flush=True)

    return cache

def main():
    print("Loading runs...", flush=True)
    b_recs, b_costs, b_sum = load_run("docs/harvest/generate_fp05_n100_baseline")
    u_recs, u_costs, u_sum = load_run("docs/harvest/generate_fp05_n100_guided")
    g_recs, g_costs, g_sum = load_run("docs/harvest/generate_fp05_n100_guided_batched")

    runs = [
        ("baseline", b_recs, b_costs, b_sum),
        ("guided_unbatched", u_recs, u_costs, u_sum),
        ("guided_batched", g_recs, g_costs, g_sum),
    ]

    report = {}

    for name, recs, costs, summary in runs:
        ph_recs = [r for r in recs if r.system == System.POST_HOC]
        ph_sum = summary["per_system"]["post_hoc"]

        n_queries = len(ph_recs)
        clean_parses = ph_sum["clean_parses"]
        clean_parse_rate = clean_parses / n_queries if n_queries else 0

        query_errors = {}
        for row in summary.get("rows", []):
            if row.get("system") == "post_hoc":
                query_errors[row["query_id"]] = row.get("errors", [])

        total_claims = 0
        valid_claims = 0
        zero_cit_claims = 0
        total_citations = 0

        for r in ph_recs:
            has_err = len(query_errors.get(r.query_id, [])) > 0
            n_c = len(r.claims)
            total_claims += n_c
            if not has_err:
                valid_claims += n_c
            for c in r.claims:
                cits = len(c.citations)
                total_citations += cits
                if cits == 0:
                    zero_cit_claims += 1

        valid_claim_parse_rate = valid_claims / total_claims if total_claims else 0
        zero_cit_pct = (zero_cit_claims / total_claims) * 100 if total_claims else 0

        s2_info = analyze_stage2(recs, costs)

        report[name] = {
            "n_queries": n_queries,
            "clean_parses": f"{clean_parses}/{n_queries}",
            "clean_parse_rate_pct": round(clean_parse_rate * 100, 2),
            "total_claims": total_claims,
            "valid_claims": f"{valid_claims}/{total_claims}",
            "valid_claim_parse_rate_pct": round(valid_claim_parse_rate * 100, 2),
            "claims_per_query": round(total_claims / n_queries, 2),
            "total_citations": total_citations,
            "zero_cit_claims": zero_cit_claims,
            "zero_cit_pct": round(zero_cit_pct, 2),
            "quote_not_found": ph_sum.get("quote_not_found", 0),
            "call_failures": ph_sum.get("call_failure_count", 0),
            "truncations_failures": sum(1 for errs in query_errors.values() if len(errs) > 0),
            "s2_call_count": s2_info["stage2_call_count"],
            "s2_prompt_tokens": s2_info["prompt_tokens"],
            "claims_per_batch": s2_info["claims_per_batch"],
        }

    print("Collecting pairs for MiniCheck...", flush=True)
    all_ph_recs = [r for name, recs, _, _ in runs for r in recs if r.system == System.POST_HOC]
    seen_pairs = {}
    def pair_collector(premise, hypothesis):
        seen_pairs.setdefault((premise, hypothesis), None)
        return False
    citation_f1(all_ph_recs, pair_collector)
    pairs = list(seen_pairs.keys())
    print(f"Total unique (premise, hypothesis) pairs to score: {len(pairs)}", flush=True)

    scores = get_cached_scores(pairs)
    phi = phi_from_scores(scores, threshold=0.5)

    for name, recs, _, _ in runs:
        ph_recs = [r for r in recs if r.system == System.POST_HOC]
        f1_res = citation_f1(ph_recs, phi)
        report[name]["citation_precision"] = round(f1_res["precision"], 4)
        report[name]["citation_recall"] = round(f1_res["recall"], 4)
        report[name]["citation_f1"] = round(f1_res["f1"], 4)

    Path("docs/harvest/three_way_comparison.json").write_text(json.dumps(report, indent=2))
    print("Saved docs/harvest/three_way_comparison.json", flush=True)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
