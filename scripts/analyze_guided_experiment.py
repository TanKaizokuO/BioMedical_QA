#!/usr/bin/env python3
import json
import statistics
from pathlib import Path
from biomedqa.schema import read_query_records, read_jsonl, CostRecord, System
from biomedqa.scoring.granularity import arm_granularity, stage_output_tokens
from biomedqa.scoring.citation import citation_f1
from biomedqa.verify import MiniCheckVerifier, score_map, phi_from_scores

def load_run(prefix_str):
    prefix = Path(prefix_str)
    rec_path = prefix.with_name(f"{prefix.name}.records.jsonl")
    cost_path = prefix.with_name(f"{prefix.name}.costs.jsonl")
    sum_path = prefix.with_name(f"{prefix.name}.summary.json")

    records = list(read_query_records(rec_path))
    costs = [CostRecord(**d) for d in read_jsonl(cost_path)]
    summary = json.loads(sum_path.read_text())

    return records, costs, summary

def analyze():
    base_recs, base_costs, base_sum = load_run("docs/harvest/generate_fp05_n100_baseline")
    guid_recs, guid_costs, guid_sum = load_run("docs/harvest/generate_fp05_n100_guided")

    base_ph_recs = [r for r in base_recs if r.system == System.POST_HOC]
    guid_ph_recs = [r for r in guid_recs if r.system == System.POST_HOC]

    # 1. Clean parse rate
    base_clean = base_sum["per_system"]["post_hoc"]["clean_parses"]
    guid_clean = guid_sum["per_system"]["post_hoc"]["clean_parses"]
    N_base = len(base_ph_recs)
    N_guid = len(guid_ph_recs)

    # 2. quote_not_found count
    base_qnf = base_sum["per_system"]["post_hoc"]["quote_not_found"]
    guid_qnf = guid_sum["per_system"]["post_hoc"]["quote_not_found"]

    # 3. Rejected call count
    base_rej = base_sum["per_system"]["post_hoc"]["call_failure_count"]
    guid_rej = guid_sum["per_system"]["post_hoc"]["call_failure_count"]

    # 4. Claim recovery rate
    base_recov_count = base_sum["per_system"]["post_hoc"]["recovered_notes"]
    guid_recov_count = guid_sum["per_system"]["post_hoc"]["recovered_notes"]

    base_total_cits = base_sum["per_system"]["post_hoc"]["total_citations"]
    guid_total_cits = guid_sum["per_system"]["post_hoc"]["total_citations"]

    base_recov_rate = (base_recov_count / base_total_cits) if base_total_cits else 0.0
    guid_recov_rate = (guid_recov_count / guid_total_cits) if guid_total_cits else 0.0

    # 5. Stage-2 prompt token count and context token count
    # Stage 2 for post-hoc has component="generate" and is the second cost record for each post-hoc query
    def stage2_stats(records, costs):
        # group costs by query_id
        by_q = {}
        for c in costs:
            if c.query_id:
                by_q.setdefault(c.query_id, []).append(c)
        
        stg2_prompts = []
        for r in records:
            if r.system == System.POST_HOC:
                q_costs = by_q.get(r.query_id, [])
                if len(q_costs) >= 2:
                    stg2_prompts.append(q_costs[1].input_tokens or 0)
        return sum(stg2_prompts), (statistics.mean(stg2_prompts) if stg2_prompts else 0)

    base_s2_tot, base_s2_mean = stage2_stats(base_recs, base_costs)
    guid_s2_tot, guid_s2_mean = stage2_stats(guid_recs, guid_costs)

    # 6. Claim-length distribution
    base_arm = arm_granularity(base_ph_recs, System.POST_HOC)
    guid_arm = arm_granularity(guid_ph_recs, System.POST_HOC)

    # Calculate token-length or word-length stats for claims
    base_claim_words = [len(c.text.split()) for r in base_ph_recs for c in r.claims]
    guid_claim_words = [len(c.text.split()) for r in guid_ph_recs for c in r.claims]

    def word_stats(words):
        if not words:
            return {"mean": 0, "median": 0, "p25": 0, "p75": 0, "p90": 0, "min": 0, "max": 0}
        s = sorted(words)
        return {
            "mean": round(statistics.fmean(s), 2),
            "median": round(statistics.median(s), 2),
            "p25": s[int(len(s)*0.25)],
            "p75": s[int(len(s)*0.75)],
            "p90": s[int(len(s)*0.90)],
            "min": s[0],
            "max": s[-1],
        }

    base_wstats = word_stats(base_claim_words)
    guid_wstats = word_stats(guid_claim_words)

    # 7. MiniCheck Citation F1
    print("Collecting pairs for MiniCheck...", flush=True)
    all_recs = base_ph_recs + guid_ph_recs
    seen_pairs = {}
    def pair_collector(premise, hypothesis):
        seen_pairs.setdefault((premise, hypothesis), None)
        return False
    citation_f1(all_recs, pair_collector)
    pairs = list(seen_pairs.keys())
    print(f"Total unique (premise, hypothesis) pairs to score: {len(pairs)}", flush=True)

    import torch
    torch.set_num_threads(8)
    verifier = MiniCheckVerifier(batch_size=64)
    scores = score_map(pairs, verifier)
    phi = phi_from_scores(scores, threshold=0.5)
    base_f1_res = citation_f1(base_ph_recs, phi)
    guid_f1_res = citation_f1(guid_ph_recs, phi)

    print("\n==================== EXPERIMENT REPORT ====================")
    print(f"Baseline N: {N_base} | Guided N: {N_guid}")
    print(f"Clean Parse Rate: Baseline = {base_clean}/{N_base} ({base_clean/N_base:.1%}) | Guided = {guid_clean}/{N_guid} ({guid_clean/N_guid:.1%})")
    print(f"quote_not_found Count: Baseline = {base_qnf} | Guided = {guid_qnf}")
    print(f"Rejected Call Count: Baseline = {base_rej} | Guided = {guid_rej}")
    print(f"Claim Recovery Rate: Baseline = {base_recov_count}/{base_total_cits} ({base_recov_rate:.1%}) | Guided = {guid_recov_count}/{guid_total_cits} ({guid_recov_rate:.1%})")
    print(f"Stage-2 Prompt Tokens: Baseline = Total {base_s2_tot} (Mean {base_s2_mean:.1f}/q) | Guided = Total {guid_s2_tot} (Mean {guid_s2_mean:.1f}/q)")
    print(f"Citation Precision: Baseline = {base_f1_res['precision']:.4f} | Guided = {guid_f1_res['precision']:.4f} (Delta: {guid_f1_res['precision']-base_f1_res['precision']:+.4f})")
    print(f"Citation Recall: Baseline = {base_f1_res['recall']:.4f} | Guided = {guid_f1_res['recall']:.4f} (Delta: {guid_f1_res['recall']-base_f1_res['recall']:+.4f})")
    print(f"Citation F1: Baseline = {base_f1_res['f1']:.4f} | Guided = {guid_f1_res['f1']:.4f} (Delta: {guid_f1_res['f1']-base_f1_res['f1']:+.4f})")
    base_cpq = base_arm.n_claims / base_arm.n_records if base_arm.n_records else 0
    guid_cpq = guid_arm.n_claims / guid_arm.n_records if guid_arm.n_records else 0
    print(f"Claims per Query: Baseline = {base_cpq:.2f} (median {base_arm.median_claims_per_query:.1f}) | Guided = {guid_cpq:.2f} (median {guid_arm.median_claims_per_query:.1f})")
    print(f"  Baseline: mean={base_wstats['mean']}, median={base_wstats['median']}, p25={base_wstats['p25']}, p75={base_wstats['p75']}, p90={base_wstats['p90']}, min={base_wstats['min']}, max={base_wstats['max']}")
    print(f"  Guided:   mean={guid_wstats['mean']}, median={guid_wstats['median']}, p25={guid_wstats['p25']}, p75={guid_wstats['p75']}, p90={guid_wstats['p90']}, min={guid_wstats['min']}, max={guid_wstats['max']}")

if __name__ == "__main__":
    analyze()
