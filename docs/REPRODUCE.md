# Reproduction workflow

This workflow is for authorized investigators with the study inputs. The public repository contains no patient data or fitted clinical models.

## Model development

Place approved CSV inputs in a private local `source_data/` directory and follow `DATA_DICTIONARY.md` exactly. Run each task with all eight algorithm families: `Logistic`, `LASSO`, `Random forest`, `XGBoost`, `ExtraTrees`, `RBF-SVM`, `ANN/MLP` and `TabPFN`.

Five outer folds estimate development performance; three inner folds select features and parameters. The external cohort is not used for model selection or threshold determination. Treatment response uses development cross-validation and independent external validation; it has no separate internal test cohort.

After all eight families finish, assemble results with:

```bash
python src/diagnostic/assemble_results.py P_vs_D P_vs_B TreatmentResponse
```

The scripts reject duplicate patient IDs, invalid labels, unknown partitions or centers, and inconsistent EANO–ESMO outcome labels. Keep generated patient-level files private.

## Fixed-model use

`configs/` records the final feature sets, parameters and thresholds. `predict.py` requires the matching approved training context and, for TabPFN, its checkpoint. It does not refit, select features or change the threshold.

Early identification uses `src/diagnostic/early_identification.py` after applying the fixed `P_vs_D` model to the independent first-negative cohort. It does not train a separate early model.

Figure scripts require the assembled intermediate source tables and saved model blocks. The public code alone cannot reproduce manuscript figures without the controlled study inputs.
