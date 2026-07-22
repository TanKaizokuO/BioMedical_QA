# Biomedical QA — Evidence-Grounded, Claim-Attributable

Research workspace for **Project 2**: a biomedical QA system that answers as a set of atomic
**claims**, each attributed to the retrieved passage(s) that support it, with a lightweight
faithfulness verifier — evaluated on attribution/faithfulness first, accuracy second.

## Layout

```
.
├── pyproject.toml, uv.lock     # uv environment (Python >=3.12,<3.14)
├── .env.example                # copy to .env and fill in keys
├── notebooks/                  # runnable companion notebooks (offline-first; toy fallbacks)
│   └── 01_1_retrieval_foundations.ipynb
├── docs/                       # planning & research notes  (git-ignored, local)
│   ├── project2_..._implementation_plan.md
│   ├── related_work.md
│   └── learning_roadmap.md
└── teach/                      # /teach study workspace       (git-ignored, local)
    ├── MISSION.md              # why we're learning this — the compass
    ├── RESOURCES.md            # curated high-trust sources
    ├── GLOSSARY.md             # canonical terminology
    ├── NOTES.md                # learner preferences / working notes
    ├── lessons/                # self-contained HTML lessons
    ├── assets/                 # shared lesson stylesheet + quiz widget
    ├── reference/              # printable cheat-sheets
    └── learning-records/       # what's been learned (steers next sessions)
```

Only the environment/config files are tracked in git; `docs/`, `teach/`, and `notebooks/` are
local study materials (see `.gitignore`).

## Getting started

```bash
uv sync                 # install the environment
uv run jupyter lab      # open the companion notebooks
```

Open a lesson directly, e.g.:

```bash
xdg-open teach/lessons/0001-retrieval-cascade-and-the-gate.html
```
