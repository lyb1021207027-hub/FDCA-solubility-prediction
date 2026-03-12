# Reproduction Guide

## Files required

Place the following files in the repository root before running the script:

- `Regression prediction code.py`
- `Validation data-3.xlsx`
- `Trained regression model.zip`

## Command

```bash
python "Regression prediction code.py"
```

## What the script does

1. Loads the trained regression model from the zip archive or model directory.
2. Reads the validation dataset.
3. Maps descriptor names to the internal model feature names.
4. Predicts the target values.
5. Computes MSE, MAE, and R2 using the true `Y` column.
6. Displays a true-vs-predicted scatter plot.
7. Exports an Excel file with predictions and residuals.

## Expected output file

- `regression_predictions_from_saved_model.xlsx`

## Troubleshooting

### Missing feature columns

If the script reports missing columns, check whether the Excel file contains the following descriptor names exactly:

- `Diffusion coefficient`
- `CN(FDCA–H)`
- `CN(FDCA–O)`
- `MEPS minimal`
- `MEPS maximal`
- `Polarity difference`
- `Solvation energy`
- `δH`
- `δT`
- `Y`

### Model loading failure

If the model cannot be loaded, confirm that `Trained regression model.zip` contains a valid AutoGluon predictor directory with `predictor.pkl`.
