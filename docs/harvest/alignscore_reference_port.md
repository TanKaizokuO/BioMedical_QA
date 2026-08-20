# AlignScore Reference Environment Port and Parity Report

## Overview

This document records the port of the AlignScore reference environment and the numerical verification of its PyTorch implementation.
AlignScore pins legacy dependencies (`torch<2`, `pytorch_lightning<2`, `protobuf<=3.20`) that conflict with the main package requirements (`torch>=2.13`).
We built an isolated environment outside the repository, retrieved the official checkpoint, and verified that a direct PyTorch port achieves exact numerical parity with the reference implementation.

## Isolated Environment Dependencies

We created the isolated environment using `uv` with Python 3.10.20 at `/tmp/alignscore_venv`.
The resolved versions of all relevant packages are:

| Package | Resolved Version | Constraint / Note |
|---|---|---|
| Python | 3.10.20 | Required for PyTorch < 2.0 |
| `torch` | 1.13.1+cu117 | `torch<2` |
| `pytorch_lightning` | 1.9.5 | `pytorch_lightning<2` |
| `protobuf` | 3.20.0 | `protobuf<=3.20` |
| `numpy` | 1.26.4 | `numpy<2` |
| `transformers` | 4.29.2 | `transformers<4.30` |
| `setuptools` | 69.5.1 | `setuptools<70` (for `pkg_resources`) |
| `nltk` | 3.9.1 | Sentence tokenization (`sent_tokenize`) |
| `spacy` | 3.8.15 | Dependency of reference `alignscore` package |
| `en_core_web_sm` | 3.8.0 | spaCy English model |
| `alignscore` | 0.1.3 | Reference implementation from official repository |

## Official Checkpoint Information

We retrieved the official ~355M parameter `AlignScore-large` checkpoint from HuggingFace.

- **Source URL**: `https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt`
- **Backbone Architecture**: RoBERTa-large (355M parameters)
- **Downloaded File Size**: 4,895,673,790 bytes
- **SHA256 Checksum**: `ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d`

## Test Pairs and Parity Scores

We executed inference on six fixed (premise, hypothesis) test pairs covering clearly entailed, clearly contradicted, and unrelated relationships.

| Pair ID | Category | Premise | Hypothesis | Reference Score | Ported Score | Absolute Difference |
|---|---|---|---|---|---|---|
| 1 | Clearly Entailed | Metformin reduces all-cause mortality in patients with type 2 diabetes. | Metformin lowers mortality risk in type 2 diabetes patients. | 0.98687267 | 0.98687267 | 0.00000000e+00 |
| 2 | Clearly Contradicted | Metformin reduces all-cause mortality in patients with type 2 diabetes. | Metformin increases mortality risk in type 2 diabetes patients. | 0.00069050 | 0.00069050 | 0.00000000e+00 |
| 3 | Unrelated | Metformin reduces all-cause mortality in patients with type 2 diabetes. | Aspirin is used to treat mild headaches. | 0.02056623 | 0.02056623 | 0.00000000e+00 |
| 4 | Clearly Entailed | Pembrolizumab plus chemotherapy improved overall survival in non-small-cell lung cancer. | Pembrolizumab combined with chemotherapy extended survival for NSCLC patients. | 0.97973013 | 0.97973013 | 0.00000000e+00 |
| 5 | Clearly Contradicted | Pembrolizumab plus chemotherapy improved overall survival in non-small-cell lung cancer. | Pembrolizumab was ineffective and caused rapid progression in all patients. | 0.00010750 | 0.00010750 | 0.00000000e+00 |
| 6 | Unrelated | Pembrolizumab plus chemotherapy improved overall survival in non-small-cell lung cancer. | Penicillin was discovered by Alexander Fleming in 1928. | 0.00652159 | 0.00652159 | 0.00000000e+00 |

## Numerical Fidelity Verification and Diagnostics

We compared the reference output vector against the ported output vector using `torch.allclose`.

- **Relative Tolerance (`rtol`)**: `1e-5`
- **Absolute Tolerance (`atol`)**: `1e-7`
- **Maximum Absolute Difference**: `0.00000000e+00`
- **Verification Verdict**: **PASS**

### Why the Difference is Exactly Zero

The comparison is between two completely separate model instances loaded from the same checkpoint weights on CPU:

1. `ref_scorer`: An instance of `alignscore.AlignScore` (which constructs `BERTAlignModel`, a `pytorch_lightning.LightningModule`).
2. `port_scorer`: An instance of `StandaloneAlignScore` (a pure `torch.nn.Module` using HuggingFace `RobertaModel` and `nn.Linear(1024, 3)`).

For each test pair, both models run independent forward passes on CPU:
- Both sentencize premise and hypothesis using `nltk.tokenize.sent_tokenize`.
- Both tokenize `(premise_chunk, hypo_sentence)` using `AutoTokenizer.from_pretrained('roberta-large')` with `truncation='only_first', padding='max_length', max_length=512`.
- Both pass the tokenized tensors through `RobertaModel` and retrieve `pooler_output`.
- Both pass `pooler_output` through the identical 3-way linear classification weights (`tri_layer`) and apply `softmax`.
- Both select index `0` (NLI Entailment probability).

Because CPU float32 matrix multiplications in PyTorch are deterministic for identical input shapes and identical weights, the two independent forward passes produce identical output floating-point numbers down to the last bit (`diff = 0.0`).

## Exact Reproduction Commands

Run these one-line commands to reproduce the isolated environment, checkpoint download, NLTK/spaCy setup, and parity check:

```bash
uv venv /tmp/alignscore_venv --python 3.10
uv pip install --python /tmp/alignscore_venv/bin/python "torch<2" "pytorch_lightning<2" "protobuf<=3.20" "numpy<2" "setuptools<70" "transformers<4.30" "nltk>=3.7"
mkdir -p /tmp/alignscore_ckpt
curl -C - -L -o /tmp/alignscore_ckpt/AlignScore-large.ckpt "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt"
git clone https://github.com/yuh-zha/AlignScore.git /tmp/alignscore_repo
uv pip install --python /tmp/alignscore_venv/bin/python /tmp/alignscore_repo
uv pip install --python /tmp/alignscore_venv/bin/python https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
/tmp/alignscore_venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
/tmp/alignscore_venv/bin/python scripts/alignscore_port.py
```
