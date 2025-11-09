# Parkinson-s-Voice-and-Clinical-Modeling
End-to-end machine learning workflow for Parkinson’s disease classification using the PPMI dataset, featuring preprocessing, UPDRS progression modeling, feature interpretation with SHAP, and robustness testing under simulated noise conditions.
Model Explainability using SHAP

To ensure transparency and interpretability of the machine learning model used for Parkinson’s disease classification, this notebook implements SHapley Additive exPlanations (SHAP). SHAP is a game-theoretic approach that quantifies each feature’s contribution to the model’s predictions.

In this project, I trained a Random Forest classifier on clinical and/or voice-derived features. To interpret the influence of individual predictors, I used the following workflow:

explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(X_test)
# For binary classification shap_vals[1] corresponds to positive class
shap.summary_plot(shap_vals[1], X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'))
print('Saved SHAP summary plot to output folder')

What this achieves

Global feature importance: Identifies which acoustic or clinical variables most strongly contribute to the model’s Parkinson’s vs control classification.

Directionality of effects: Shows whether increases in a feature raise or lower PD probability.

Reproducible documentation: The summary plot is saved to the output directory (shap_summary.png), allowing transparent reporting of the model’s behavior in the repository.

This aligns with current recommendations in medical machine learning, where explainability is essential for clinical trust, regulatory considerations, and scientific interpretability.

Robustness Testing via Noise Perturbation

To assess how well the trained model generalizes to imperfect or noisy data, I implemented an initial robustness evaluation framework. The goal is to simulate real-world variability — such as sensor noise or recording artifacts — and measure how much model performance degrades under controlled perturbations.

The following function introduces Gaussian noise based on a specified signal-to-noise ratio (SNR):

def evaluate_robustness_numeric(clf, X_test, y_test, noise_snr_db=20):
    """Add Gaussian noise scaled by desired SNR (dB) to numeric features and eval model.
    This is a placeholder — for audio-level noise you should perturb audio files before
    extracting features in R or Python audio pipelines.
    """
    Xn = X_test.copy()
    rms_signal = np.sqrt((Xn ** 2).mean())
    snr_lin = 10 ** (noise_snr_db / 10.0)
    noise_std = np.sqrt(rms_signal / snr_lin)
    noise = np.random.normal(0, noise_std, size=Xn.shape)
    Xn += noise
    y_pred = clf.predict(Xn)
    acc = accuracy_score(y_test, y_pred)
    print(f'Robustness test at SNR={noise_snr_db} dB: Accuracy = {acc:.4f}')

Why this matters

Medical voice datasets often contain real-world noise (background speech, microphone quality differences, device variability).

Robustness testing evaluates whether the classifier is overly sensitive to perturbations.

This step encourages development of models that are clinically reliable, not just statistically accurate.

Forms the foundation for future work (e.g., adversarial robustness, domain shift evaluation).

In current machine learning literature, robustness evaluation is considered a core component of responsible deployment in healthcare contexts.

Reproducibility & Data Traceability

To support reproducibility, the notebook saves intermediate outputs and diagnostic tables.
This ensures transparency in how data were filtered, cleaned, and prepared.

Example:

if participant_master is not None:
    out_fp = os.path.join(OUTPUT_DIR, 'participant_master_sample.csv')
    participant_master.head(200).to_csv(out_fp, index=False)
    print('Saved participant_master sample to', out_fp)

Purpose

Provides a verifiable record of the preprocessing pipeline.

Allows collaborators and reviewers to inspect input data formats.

Aligns with open-science principles recommended for clinical machine learning research.
