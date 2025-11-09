# Parkinson’s Voice and Clinical Modeling

**An end-to-end machine learning workflow for Parkinson’s disease classification using the PPMI dataset.** This project integrates data preprocessing, UPDRS progression modeling, model training, feature interpretation with SHAP, and robustness testing under simulated noise conditions. The repository is designed to be clear, accessible, and portfolio-friendly while still reflecting best practices in medical machine learning.

---

##  Model Explainability with SHAP

To understand how the model makes predictions, this project incorporates **SHapley Additive exPlanations (SHAP)**. SHAP helps identify how each feature contributes to the model’s Parkinson’s vs. control classification.

### How It Works

A Random Forest classifier is trained on clinical and/or voice-derived features. SHAP is then used to compute feature importance values:

```python
explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_test)
shap.summary_plot(shap_vals[1], X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'))
```

### What This Provides

* **Global feature importance** — identifies which features most strongly influence predictions.
* **Directionality of effects** — shows whether higher values increase or decrease PD likelihood.
* **Reproducible documentation** — SHAP plots are automatically saved for transparency.

Explainability is essential in medical ML, making this a valuable addition for trust, future clinical use, and professional presentation.

---

##  Robustness Testing with Noise Perturbation

Real-world clinical and voice data often contain noise from sensors, recording conditions, or environmental factors. This project includes a first-step robustness test by adding Gaussian noise to input features and re-evaluating model performance.

### Method

```python
def evaluate_robustness_numeric(clf, X_test, y_test, noise_snr_db=20):
    Xn = X_test.copy()
    rms_signal = np.sqrt((Xn ** 2).mean())
    snr_lin = 10 ** (noise_snr_db / 10.0)
    noise_std = np.sqrt(rms_signal / snr_lin)
    noise = np.random.normal(0, noise_std, size=Xn.shape)
    Xn += noise
    y_pred = clf.predict(Xn)
    acc = accuracy_score(y_test, y_pred)
    print(f'Robustness test at SNR={noise_snr_db} dB: Accuracy = {acc:.4f}')
```

### Why It Matters

* Simulates noisy recordings or device variability.
* Tests if the classifier is stable under perturbations.
* Encourages development of **clinically reliable**, not just statistically strong, models.
* Forms a baseline for future robustness and adversarial tests.

Robustness is increasingly recognized as a requirement for responsible healthcare AI.

---

##  Reproducibility & Data Traceability

To ensure full transparency and reproducibility, the notebook saves intermediate outputs, summary files, and diagnostic data.

### Example

```python
if participant_master is not None:
    out_fp = os.path.join(OUTPUT_DIR, 'participant_master_sample.csv')
    participant_master.head(200).to_csv(out_fp, index=False)
    print('Saved participant_master sample to', out_fp)
```

### Why This Step Is Important

* Creates a clear record of preprocessing decisions.
* Allows other researchers or collaborators to inspect intermediate stages.
* Aligns with best practices in open, traceable scientific workflows.

---



It is designed to be both a practical tool and a strong demonstration of your applied machine learning and biomedical data analysis skills.

