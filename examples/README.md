# Synthetic example data

`diagnostic_data.csv` contains 40 entirely synthetic rows with the same column names, data types and partition structure used by the diagnostic training code. IDs start with `SYNTHETIC_`; no row represents a patient.

Run from the repository root:

```bash
python src/diagnostic/nested_analysis.py --input examples/diagnostic_data.csv --output examples/output --task P_vs_D --family Logistic
```

This command demonstrates data reading, preprocessing, nested feature/parameter selection and evaluation. The small cohort is for a software smoke demonstration only. Its metrics must not be interpreted as study results or clinical accuracy. The eight algorithm families and full manuscript results require the authorized study cohort.
