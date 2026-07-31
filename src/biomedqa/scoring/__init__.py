"""Pure functions over the frozen schema. Nothing here reads a model, a corpus, or the network.

That purity is the point: every table is recomputable from `runs/*/records.jsonl` without rerunning
anything. Re-chunking, a new threshold, a different k, or the hit@10 fallback in G1's escalation
ladder are all re-*scores*, not re-*runs* — which only holds while nothing upstream binarizes.

    retrieval.py     hit@k, recall@k, MRR, nDCG, Wilson intervals   → Table 1
    citation.py      ALCE citation precision / recall / F1          → Table 2
    calibration.py   AUROC, ECE, threshold sweeps                   → Table 3
    cost.py          tokens, USD, wall-clock per query              → Table 4
    strata.py        negation / numerics / scope error analysis     → Table 5
    agreement.py     Krippendorff's α, binary collapse and 4-way    → G4
    accuracy.py      PubMedQA yes/no/maybe (secondary, C6)
"""
