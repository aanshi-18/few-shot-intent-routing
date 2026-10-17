# Few-shot support routing and unknown-intent detection

I compared two intent-routing approaches on BANKING77: TF-IDF with logistic
regression, and frozen MiniLM sentence embeddings with category prototypes.
The study asks how well a system learns newly added support categories from a
few fitting examples, and whether it can still reject unsupported requests.

## Results

| Model | Fitting examples per new category | New-intent accuracy | Supported macro-F1 | Unknown false acceptance |
|---|---:|---:|---:|---:|
| TF-IDF + LR | 1 | 0.35% | 0.603 | 69.97% |
| TF-IDF + LR | 5 | 11.93% | 0.656 | 69.31% |
| TF-IDF + LR | 10 | 30.78% | 0.731 | 66.50% |
| MiniLM prototypes | 1 | 41.94% | 0.749 | 73.92% |
| MiniLM prototypes | 5 | 75.70% | 0.847 | 69.25% |
| MiniLM prototypes | 10 | 79.91% | 0.856 | 67.86% |

Values average sampling seeds within each category split, then weight the three
category splits equally. Prototypes adapt much better in this study. Neither
method reliably rejects unfamiliar requests at the selected high-coverage
operating point. The comparison is between two fixed practical pipelines,
not a claim of a new algorithm or a production-ready support system.

## Protocol

- Category allocation: 45 original, 15 new, seven development-only unknown,
  ten final-test unknown categories.
- Three category splits and three adaptation samples; nested 1/5/10 examples.
- Seven train/test text-overlap rows excluded before learning/dev partitions.
  Raw CSVs and the official test set are preserved.
- Additional labelled development messages from new categories calibrate rejection.
  This is **few-shot fitting with extra calibration data**, not strict k-shot
  end-to-end learning.
- Accept if maximum class probability or maximum cosine exceeds the lower
  10th-percentile supported-development score. Recalibration uses permitted
  development messages only; no test-driven threshold changes.
- Per-category F1 includes errors across all supported messages. Rejection-aware
  correct routing and unknown false acceptance are separate measures.
- 54 post-adaptation configurations plus six unique initial baselines.
  Split variation is descriptive: overlapping splits are not independent datasets.

## Files

- `run_experiment.py`: dataset checks, splits, original LR pilot.
- `continue_project.py`: corrected metrics, encoder cache and prototype pilot.
- `full_study.py`: checkpointed repeated experiments and summaries.
- `finish_report.py`: figures, deterministic error sample, report and CV material.
- `results/full_study/`: manifests, prediction logs, development scores and summaries.
- `report/Banking77_Research_Report.pdf` and `report/REPORT.md`: five-page report
  and editable text; `error_audit.csv` contains 42 reviewed model-message cases.
- `CV_BULLETS.md` and `INTERVIEW_GUIDE.md`: separate application material.
- `PROJECT_STATE.md`: completed work and handoff; historical checkpoints are retained.

## Running

Use Python 3.12 in a fresh virtual environment. `requirements-stage4.txt` records
Stage 4 tested dependencies; Stage 5 environment is recorded separately in
`results/full_study/environment.json`. The CPU torch wheel may need installation
from https://download.pytorch.org/whl/cpu before installing the requirements.
Report generation additionally uses `requirements-report.txt`.

```bash
python -m pip install -r requirements-stage4.txt
python -m pip install -r requirements-report.txt
python full_study.py check
python full_study.py summarize
python finish_report.py
```

To resume experiments, use `python full_study.py run`. Verified checkpoints are
reused. Source/config/data mismatches cause errors rather than silently accepting
stale results. Existing cached embeddings mean no download or encoder inference
is needed for the repeated study. To rebuild the cache, use
`python continue_project.py embed`; this requires downloading the pinned encoder.

## Limitations

New categories receive extra calibration labels. LR is unweighted and faces
class imbalance, whereas MiniLM benefits from external pretraining. These factors
have not been isolated. The task uses artificial intent holdouts in one English
banking dataset. Encoder pretraining contamination was not audited. Pilot LR
predictions were reused from a different scikit-learn version; provenance is saved.
Prototype timings use cached embeddings and are not end-to-end latency claims.

The error audit is AI-assisted and illustrative, not an independently adjudicated
or representative estimate of failure frequencies. Code and report preparation
were AI-assisted. Test labels were not changed following the audit.

## Data and references

BANKING77: https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data
(CC BY 4.0; licence retained in data/raw/LICENSE).

- Casanueva et al. 2020: https://aclanthology.org/2020.nlp4convai-1.5/
- Larson et al. 2019: https://aclanthology.org/D19-1131/
- Snell et al. 2017: https://arxiv.org/abs/1703.05175
- Encoder: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Full model identity and methodological details are in the report.
