# CSF-LM: Diagnosis and Treatment Response Prediction

This repository accompanies a two-center study of routine cerebrospinal fluid measurements for diagnosing leptomeningeal metastasis and predicting treatment response. It contains two diagnostic tasks and one EANO–ESMO treatment-response task. The code implements eight algorithms, nested cross-validation, feature selection, model evaluation and manuscript figure generation. Early identification uses the fixed primary diagnostic model for initially cytology-negative patients.

## Contents

- `src/diagnostic/`: nested training, result assembly, explanations and fixed-model early-identification summary.
- `src/figures/`: manuscript cohort and figure-generation code.
- `configs/`: final feature sets, parameters and thresholds for `P_vs_D`, `P_vs_B` and `TreatmentResponse`.
- `predict.py`: fixed-model inference entry point; it requires separately supplied model assets.
- `docs/`: input definitions and the controlled-access reproduction workflow.
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT license for project code only.

## Tasks

- `P_vs_D`: leptomeningeal metastasis versus cancer without leptomeningeal metastasis.
- `P_vs_B`: leptomeningeal metastasis versus benign controls.
- `TreatmentResponse`: responder (1) versus non-responder (0) under EANO–ESMO criteria; stable and progressive disease are non-response.

The predictors are routine CSF measurements: total protein, glucose, chloride ion, lactate, LDH, ADA, AST, nucleated cell count, total cell count, RBC count, LNR and LMR.

Exact manuscript reproduction requires the approved study inputs and matching model assets. Third-party software and pretrained weights retain their own licenses.



## Research use

This repository is provided for research and reproducibility purposes only and is not intended for clinical diagnosis or treatment decision-making.

## Manuscript figure mapping

| Manuscript figure | Code entry point |
|---|---|
| Figure 1 | `src/figures/plot_cohorts.py` |
| Figure 2 | `src/figures/plot_current_manuscript.py --figure Figure2` |
| Figure 3 | `src/figures/plot_current_manuscript.py --figure Figure3` |
| Figure 4 | `src/figures/plot_current_manuscript.py --figure Figure4` |
| Supplementary Figures | `src/figures/plot_current_manuscript.py --figure SFigure1` through `SFigure7` |
