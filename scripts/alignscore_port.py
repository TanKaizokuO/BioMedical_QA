"""AlignScore reference environment port and numerical parity harness.

This script executes the AlignScore reference implementation alongside a
standalone PyTorch port that loads model weights directly from the Lightning
checkpoint without needing pytorch_lightning or legacy protobuf dependencies.

Requirements pinned in isolated venv:
  torch: 1.13.1+cu117
  pytorch_lightning: 1.9.5
  protobuf: 3.20.0
  numpy: 1.26.4
  transformers: 4.29.2
"""

import argparse
import sys
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, RobertaModel


class StandaloneAlignScore(nn.Module):
    """Standalone PyTorch implementation of AlignScore-large (355M params)."""

    def __init__(self, model_name: str = "roberta-large") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = RobertaModel.from_pretrained(model_name)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.softmax = nn.Softmax(dim=-1)

    def load_from_checkpoint(self, ckpt_path: str) -> None:
        """Load state dict from Lightning checkpoint without pytorch_lightning."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]
        base_sd = {
            k.replace("base_model.", ""): v
            for k, v in sd.items()
            if k.startswith("base_model.")
        }
        tri_sd = {
            k.replace("tri_layer.", ""): v
            for k, v in sd.items()
            if k.startswith("tri_layer.")
        }
        self.base_model.load_state_dict(base_sd, strict=False)
        self.tri_layer.load_state_dict(tri_sd)
        self.eval()

    def score_pair(self, premise: str, hypo: str) -> float:
        """Score factual consistency of premise-hypothesis pair."""
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield " ".join(lst[i : i + n])

        premise_sents = sent_tokenize(premise) or [""]
        n_chunk = len(premise.strip().split()) // 350 + 1
        n_chunk = max(len(premise_sents) // n_chunk, 1)
        premise_sents = [c for c in chunks(premise_sents, n_chunk)]

        hypo_sents = sent_tokenize(hypo) or [""]

        premise_m, hypo_m = [], []
        for p in premise_sents:
            for h in hypo_sents:
                premise_m.append(p)
                hypo_m.append(h)

        inputs = self.tokenizer(
            premise_m,
            hypo_m,
            truncation="only_first",
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            out = self.base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            logits = self.tri_layer(out.pooler_output)
            probs = self.softmax(logits)
            # Index 0 corresponds to NLI Entailment
            entail_scores = probs[:, 0]

        entail_matrix = entail_scores.view(len(premise_sents), len(hypo_sents))
        score = entail_matrix.max(dim=0).values.mean().item()
        return float(score)


TEST_PAIRS = [
    {
        "category": "clearly_entailed",
        "premise": "Metformin reduces all-cause mortality in patients with type 2 diabetes.",
        "hypothesis": "Metformin lowers mortality risk in type 2 diabetes patients.",
    },
    {
        "category": "clearly_contradicted",
        "premise": "Metformin reduces all-cause mortality in patients with type 2 diabetes.",
        "hypothesis": "Metformin increases mortality risk in type 2 diabetes patients.",
    },
    {
        "category": "unrelated",
        "premise": "Metformin reduces all-cause mortality in patients with type 2 diabetes.",
        "hypothesis": "Aspirin is used to treat mild headaches.",
    },
    {
        "category": "clearly_entailed",
        "premise": "Pembrolizumab plus chemotherapy improved overall survival in non-small-cell lung cancer.",
        "hypothesis": "Pembrolizumab combined with chemotherapy extended survival for NSCLC patients.",
    },
    {
        "category": "clearly_contradicted",
        "premise": "Pembrolizumab plus chemotherapy improved overall survival in non-small-cell lung cancer.",
        "hypothesis": "Pembrolizumab was ineffective and caused rapid progression in all patients.",
    },
    {
        "category": "unrelated",
        "premise": "Pembrolizumab plus chemotherapy improved overall survival in non-small-cell lung cancer.",
        "hypothesis": "Penicillin was discovered by Alexander Fleming in 1928.",
    },
]


def run_parity_check(ckpt_path: str) -> None:
    from alignscore import AlignScore

    print(f"Loading reference AlignScore from {ckpt_path}...")
    ref_scorer = AlignScore(
        model="roberta-large",
        batch_size=2,
        device="cpu",
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
    )

    print("Loading Standalone PyTorch Port...")
    port_scorer = StandaloneAlignScore("roberta-large")
    port_scorer.load_from_checkpoint(ckpt_path)

    ref_scores = []
    port_scores = []

    print("\n--- Running Parity Check ---")
    for i, item in enumerate(TEST_PAIRS):
        p, h = item["premise"], item["hypothesis"]
        r_score = ref_scorer.score(contexts=[p], claims=[h])[0]
        p_score = port_scorer.score_pair(p, h)
        ref_scores.append(r_score)
        port_scores.append(p_score)
        print(f"Pair {i+1} [{item['category']}]:")
        print(f"  Premise:    {p}")
        print(f"  Hypothesis: {h}")
        print(f"  Ref Score:  {r_score:.8f}")
        print(f"  Port Score: {p_score:.8f}")
        print(f"  Diff:       {abs(r_score - p_score):.8e}\n")

    ref_tensor = torch.tensor(ref_scores, dtype=torch.float64)
    port_tensor = torch.tensor(port_scores, dtype=torch.float64)

    rtol = 1e-5
    atol = 1e-7
    passed = torch.allclose(ref_tensor, port_tensor, rtol=rtol, atol=atol)
    max_diff = (ref_tensor - port_tensor).abs().max().item()

    print("=== Parity Check Result ===")
    print(f"Max Absolute Difference: {max_diff:.8e}")
    print(f"allclose(rtol={rtol}, atol={atol}): {'PASS' if passed else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AlignScore Parity Check")
    parser.add_argument(
        "--ckpt-path",
        default="/tmp/alignscore_ckpt/AlignScore-large.ckpt",
        help="Path to AlignScore-large checkpoint",
    )
    args = parser.parse_args()
    run_parity_check(args.ckpt_path)


if __name__ == "__main__":
    main()
