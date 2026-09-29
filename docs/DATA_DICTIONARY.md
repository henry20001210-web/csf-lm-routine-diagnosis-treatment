# Study input definitions

The following fields are required when controlled study data are supplied locally. No patient data are included in this repository.

| Field | Definition | Unit or coding |
|---|---|---|
| `患者ID` | Pseudonymous patient identifier | Unique; no names or record numbers |
| `CSF protein` | Total CSF protein | g/L |
| `Glucose` | CSF glucose | mmol/L |
| `Chloride` | Chloride ion | mmol/L |
| `Lactate` | CSF lactate | mmol/L |
| `LDH` | Lactate dehydrogenase | U/L |
| `ADA` | Adenosine deaminase | U/L |
| `AST` | Aspartate aminotransferase | U/L |
| `Nucleated cells` | Nucleated cell count | 10⁶/L |
| `Total cells` | Total cell count | 10⁶/L |
| `RBC` | Red blood cell count | 10⁶/L |
| `LNR` | Lymphocyte-to-neutrophil ratio | Dimensionless |
| `LMR` | Lymphocyte-to-monocyte ratio | Dimensionless |

## Diagnostic inputs

`分组` is `阳性组` for LM, `癌症非脑膜转移对照组` for cancer controls, or `阴性组` for benign controls. `y` is 1 for LM and 0 for controls. `partition` is `Development`, `Internal test`, or `Nanfang-sheet validation`. `患者ID` must be unique.

## Treatment-response inputs

`centre` is `Sanjiu` for development or `Nanfang` for external validation. `response_status` is `Responder`, `Stable disease`, or `Progressive disease`; `y` is 1 only for responder and 0 for the other two categories. Measurements are obtained at treatment initiation, and response is assessed after two cycles using EANO–ESMO criteria.

## Early-identification inputs

The independent first-negative cohort contains `source` (`Sanjiu` or `Nanfang-sheet`) and `首次阴性至首次阳性天数` (1–180 days). It is evaluated using the fixed primary `P_vs_D` model and is not used to train a separate early-identification model.

Missing laboratory values remain missing; they must not be replaced with zero. Ratio values with a zero or missing denominator remain missing. Values must be numeric, nonnegative and expressed in the units above.
