# Exoplanet Candidate Classification — KOI + TOI + K2

**Short version:** this repo trains a few classic ML models (no GPUs needed) to tell apart real exoplanets from false positives in three catalogs: **KOI** (Kepler), **TOI** (TESS) and **K2**.  
Each catalog gets its own preprocessing + models. Then we **calibrate** probabilities and do a small **meta‑vote** across catalogs to produce one final prediction per object.

If you just want to try it: jump to **[Quickstart](#quickstart)**.

---

## Why this exists

Vetters and pipelines keep evolving, but sometimes you just want a **plain, reproducible baseline** you can run on a laptop and point at a CSV. This project is that: simple features, scikit‑learn + XGBoost, sensible defaults, and all artifacts saved to disk.

---

## What’s inside

```
KOI + TOI + K2/
├─ k2_run/   # K2 data → preprocessing, CPU models, calibrated voter
├─ koi_run/  # KOI data → preprocessing, CPU models, calibrated voter
├─ toi_run/  # TOI data → preprocessing, CPU models, calibrated voter
├─ multi_stage_voting.py                # step2: build voters; step3: cross‑domain meta‑vote; infer
├─ make_final_samples_csv_koi_toi_k2.py # helper to align rows across catalogs
├─ meta_run/                            # meta‑vote artifacts (created later)
├─ final_samples_for_meta.csv           # example aligned table
└─ preds_final.csv                      # example predictions
```

Key ideas in plain English:
- Each catalog uses a **fixed feature set** and a clean ETL (impute, scale, drop constants).
- We train **Random Forest**, **XGBoost** and a small **stacked** model per catalog (all on CPU).
- We **calibrate** scores (isotonic) and pick a **decision threshold** that maximizes F1.
- Finally, we **combine** KOI/TOI/K2 calibrated probabilities into a single “final vote”.

---

## Features per catalog (kept small and physical)

- **KOI:** koi_period, koi_duration, koi_depth, koi_prad, koi_insol, koi_teq
- **TOI:** pl_orbper, pl_trandurh, pl_trandep, pl_rade, pl_insol, pl_eqt, st_teff, st_logg
- **K2:**  pl_orbper, pl_trandur, pl_trandep, pl_rade, pl_insol, pl_eqt, st_teff, st_logg

The exact lists are written to `*/preprocessed/feature_names.json` after you run preprocessing.

---

## Numbers you can expect (held‑out test, using the learned vote threshold)

- **K2:** F1 **0.994**, Precision 0.994, Recall 0.994, AUROC 0.998 (thr ≈ 0.45)
- **KOI:** F1 **0.861**, Precision 0.818, Recall 0.908, AUROC 0.96 (thr ≈ 0.4)
- **TOI:** F1 **0.823**, Precision 0.743, Recall 0.921, AUROC 0.89 (thr ≈ 0.325)

Full details live in `*/voter/metrics.json`.

---

## Quickstart

> Requirements: Python 3.9+, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `joblib` (see [Environment](#environment)).

1) **Preprocess** each catalog (uses the sample CSVs included in the repo):
```bash
# KOI
python "koi_run/koi_preprocess_fixed.py" --csv "koi_run/KOI_base.csv" --out_dir "koi_run/preprocessed" --save_csv

# TOI
python "toi_run/toi_preprocess.py"       --csv "toi_run/TOI_base.csv" --out_dir "toi_run/preprocessed" --save_csv

# K2
python "k2_run/k2_preprocess.py"         --csv "k2_run/K2_base.csv"   --out_dir "k2_run/preprocessed" --save_csv
```

2) **Train** the CPU models per catalog:
```bash
python "koi_run/koi_cpu_fixed.py" --data_npz "koi_run/preprocessed/data_ready.npz" --model_dir "koi_run/models_cpu"
python "toi_run/toi_cpu.py"       --data_npz "toi_run/preprocessed/data_ready.npz" --model_dir "toi_run/models_cpu"
python "k2_run/k2_cpu.py"         --data_npz "k2_run/preprocessed/data_ready.npz"  --model_dir "k2_run/models_cpu"
```

3) Build the **calibrated local voters** (one per catalog):
```bash
python "multi_stage_voting.py" step2 --run_dir "koi_run" --domain KOI
python "multi_stage_voting.py" step2 --run_dir "toi_run" --domain TOI
python "multi_stage_voting.py" step2 --run_dir "k2_run"  --domain K2
```

4) Create the **cross‑catalog meta‑vote**:
```bash
# Weighted soft‑vote with small‑n temperature + weight capping
python "multi_stage_voting.py" step3   --domains "koi_run" "toi_run" "k2_run"   --out_dir "meta_run"   --mode "soft-temp"   --alpha 0.5   --cap_ratio 1.75   --small_n 400   --temp_small 1.5

# Or train a tiny logistic regression meta‑model instead of fixed weights:
# python "multi_stage_voting.py" step3 --domains "koi_run" "toi_run" "k2_run" --out_dir "meta_run" --use_logreg
```

5) **Run inference** on your own CSV:
```bash
python "multi_stage_voting.py" infer   --domains "koi_run" "toi_run" "k2_run"   --final_dir "meta_run"   --input_csv "path/to/your_candidates.csv"   --out_csv "preds_final.csv"
```

What you’ll get in the output CSV:
- Per‑catalog probabilities and hard calls: `p_koi`, `pred_koi`, `p_toi`, `pred_toi`, `p_k2`, `pred_k2`
- Final combo: `p_final`, `pred_final`

---

## How it works (without jargon)

1. **Clean the tables** → pick a small set of columns per catalog; fill holes; scale things so models don’t get confused by units; remove columns that don’t vary.
2. **Train three simple models** → RF, XGB, and a small stack that learns from both.
3. **Make scores meaningful** → calibrate so “0.7” is actually “~70% chance” on validation.
4. **Pick a sensible cutoff** → try a few thresholds and keep the one with best F1.
5. **Let catalogs vote** → turn the three per‑catalog probabilities into one final score.

No GPUs, no heavy deep nets—just strong, boring baselines you can trust.

---

## Artifacts you’ll see

- `*/models_cpu/` — fitted RF/XGB/Stack models + a `cpu_results_meta.json` with the context.
- `*/preprocessed/` — `data_ready.npz`, `feature_names.json`, and the preprocessing objects.
- `*/voter/` — the calibrated “local voter”: `models.pkl`, `calibrator_isotonic.pkl`, `voter_meta.json`, `metrics.json`.
- `meta_run/` — `final_vote_meta.pkl` (and, if chosen, `final_vote_lr.pkl`).

Everything is plain files, so you can diff/track them in git or ship them elsewhere.

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn xgboost joblib
```

If you prefer a file, drop those lines into `requirements.txt`.

---

## Troubleshooting

- **“Missing columns”** → check the feature lists above; your CSV must include the right column names per catalog.
- **“Everything predicts one class”** → your data may be too imbalanced; try adjusting the decision threshold in `voter_meta.json` or retrain with more negatives/positives.
- **“Why is K2 so strong?”** → K2 in this setup happens to be very clean; see the metrics above.
- **Windows path issues** → wrap paths in quotes, especially the `KOI + TOI + K2` folder.

---

## FAQ

**Do I need all three catalogs to run inference?**  
No. If your CSV only has KOI‑like columns, we use that voter and skip the rest. The final vote is computed from whatever is available.

**Can I change the final combiner?**  
Yes. Use `--use_logreg` in step 3 to train a tiny logistic regression meta‑model on top of the calibrated probabilities.

**Where do the labels come from?**  
From the catalogs’ disposition fields after a simple mapping (confirmed/planet vs false positive). Rows with “candidate/other” are dropped during preprocessing.

---

## License

Pick a license that matches how you want others to use this (MIT/Apache‑2.0 are popular). If you’re unsure, https://choosealicense.com is a great guide.
