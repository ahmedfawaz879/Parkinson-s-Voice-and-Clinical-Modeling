# Parkinson's Voice and Clinical Modeling

A PPMI clinical-data pipeline for Parkinson's disease (PD) classification and UPDRS progression modeling, with SHAP/LIME explainability and a feature-level noise-robustness test.

## Clinical problem

Parkinson's disease is diagnosed and monitored largely through clinical exams and patient-reported questionnaires (e.g. MDS-UPDRS), which are variable across raters and visits. Two modeling tasks are addressed here:

- **PD classification**: distinguishing PD patients from controls using clinical/demographic (and, upstream, potentially voice-derived acoustic) features. Model-assisted screening could help triage patients toward specialist evaluation, but any such use requires far more validation than exists in this repo (see [Limitations](#limitations)).
- **UPDRS progression modeling**: predicting a participant's total motor/non-motor symptom score (MDS-UPDRS `TOTAL`) from other measured features, as a step toward modeling disease trajectory over time. Progression modeling matters clinically because it could help anticipate care needs and evaluate whether interventions slow decline.

Explainability (SHAP, LIME) and robustness testing are included because a clinical-facing model that cannot be inspected or shown to degrade predictably under noise is not something a clinician or reviewer should trust on its accuracy numbers alone.

## Dataset

This pipeline is built against the **[Parkinson's Progression Markers Initiative (PPMI)](https://www.ppmi-info.org/)** dataset. PPMI data is **gated**: accessing it requires a study application and a signed **Data Use Agreement**, submitted through the official portal:

**https://www.ppmi-info.org/access-data-specimens/download-data**

PPMI data is **not bundled with this repository** and cannot be redistributed. `src/parkinsons_voice/data.py` documents the expected local layout (`./PPMI_raw/` for raw CSV exports, `./processed/` for cleaned tables) and raises a clear `PPMIDataNotFoundError` if that data is not present locally - there is no downloader here, by design.

## Method

- **Preprocessing** (`features.py`): median imputation + standard scaling of numeric clinical features; SMOTE-based class-imbalance handling (`imbalanced-learn`) applied to the training split only.
- **PD classifier** (`models.py`, `train.py`): `RandomForestClassifier(n_estimators=200, random_state=0)`, trained/evaluated via a stratified train/test split.
- **UPDRS regressor** (`models.py`, `train.py`): `RandomForestRegressor` predicting the MDS-UPDRS `TOTAL` score from available numeric features.
- **Explainability** (`explain.py`): SHAP `TreeExplainer` summary plots for global feature importance, and a LIME `LimeTabularExplainer` for per-instance explanations.
- **Robustness testing** (`robustness.py`): a feature-level Gaussian-noise perturbation sweep across configurable SNR levels, re-evaluating classifier accuracy at each level. **This is feature-space noise, not audio-level noise** - it perturbs already-extracted numeric features, not raw audio recordings. A true audio-level robustness test (perturbing waveforms before feature extraction) is out of scope; no audio pipeline exists in this repo.
- **Configuration** (`config.py`, `configs/default.yaml`): all hyperparameters (`n_estimators`, `test_size`, `random_state`, SNR levels, etc.) are centralized in one YAML file rather than duplicated as magic numbers across modules; a single `seed` propagates to numpy and all estimators.

## Results

**Implementation only; not yet evaluated on benchmark data.** This pipeline has never been run end-to-end against real PPMI data - PPMI access requires an approved application and Data Use Agreement (see [Dataset](#dataset)), which had not been completed at the time of writing. No accuracy, F1, AUC, or other performance numbers are reported here, because none have been legitimately computed on real clinical data. `tests/` uses small synthetic dataframes to confirm the code runs correctly (shapes, monotonic-ish robustness degradation, no exceptions) - those are correctness checks, not performance claims.

## Limitations

- **Never run against real PPMI data.** All code has only been exercised against small synthetic dataframes in `tests/`.
- **Robustness testing is feature-level only**, not audio-level. See the [Method](#method) section.
- **Single baseline architecture.** Only a `RandomForestClassifier`/`RandomForestRegressor` pair is implemented; no comparison against other model families (gradient boosting, linear baselines, neural nets) exists.
- **No external validation cohort.** Even once real PPMI data is used, results would reflect a single-site/single-study train/test split, not generalization to an independent cohort.
- **The "voice" pipeline never merges voice features into the classifier input in this reconstruction** - it faithfully preserves the original notebook's structure, where a `voice_features` table is loaded but not joined into the clinical modeling table anywhere in the code. If your upstream data prep joins voice-derived features into `clinical_clean.csv` before this pipeline runs, this is not an issue; otherwise, voice-derived signal is not currently reaching the classifier.
- **LIME and SMOTE/imbalanced-learn were dependencies-in-name-only in the source material** this repo was reconstructed from; the implementations here are new code following the same design pattern as the SHAP/classifier code, not literal recoveries. See inline module docstrings for details.

## Reproduce

```bash
git clone https://github.com/ahmedfawaz879/Parkinson-s-Voice-and-Clinical-Modeling.git
cd Parkinson-s-Voice-and-Clinical-Modeling
pip install -r requirements.txt
pytest tests/ -v
```

The three commands above run the full test suite against synthetic data and require no PPMI access. To run the pipeline itself against real data, place processed PPMI tables under `./processed/` (see [Dataset](#dataset)) and run:

```bash
python main.py train --updrs
python main.py robustness
```

(`python main.py --help` describes all options; both commands also work as the `parkinsons-voice` console script after `pip install -e .`.)

## Citation

If you use this code, please cite the repository:

```bibtex
@software{fawaz2026parkinsons,
  author = {Fawaz, Ahmed},
  title = {Parkinson's Voice and Clinical Modeling},
  year = {2026},
  url = {https://github.com/ahmedfawaz879/Parkinson-s-Voice-and-Clinical-Modeling}
}
```

If you use PPMI data, please cite PPMI per their data-use requirements:

```bibtex
@misc{ppmi,
  title = {Parkinson's Progression Markers Initiative (PPMI)},
  author = {{Parkinson's Progression Markers Initiative}},
  url = {https://www.ppmi-info.org/},
  note = {Data used under a PPMI Data Use Agreement}
}
```

## License

MIT - see [LICENSE](LICENSE).
