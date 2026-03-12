# FDCA Solubility Regression Prediction

This repository contains the code and supplementary files used to run saved-model prediction for the small-sample FDCA solubility regression task based on a trained TabPFN/AutoGluon model.

## Included project files

The repository is designed to work with the following core files in the project root:

- `Regression prediction code.py` — prediction script
- `Validation data-3.xlsx` — validation dataset with 9 descriptor columns and 1 true target column (`Y`)
- `Trained regression model.zip` — compressed trained regression model

## Recommended repository structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── Regression prediction code.py
├── Validation data-3.xlsx
├── Trained regression model.zip
├── docs/
│   ├── data_dictionary.md
│   └── reproduction_guide.md
└── results/
    └── .gitkeep
```

## Input columns expected in `Validation data-3.xlsx`

The validation file should contain the following columns:

1. `Diffusion coefficient`
2. `CN(FDCA–H)`
3. `CN(FDCA–O)`
4. `MEPS minimal`
5. `MEPS maximal`
6. `Polarity difference`
7. `Solvation energy`
8. `δH`
9. `δT`
10. `Y`

The script internally maps the first nine descriptor columns to the model feature names `f0`–`f8` and uses `Y` as the true target value for evaluation.

## Quick start

### 1. Create an environment

Using pip:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate fdca-solubility
```

### 2. Run prediction

```bash
python "Regression prediction code.py"
```

After execution, the script will:

- load the trained model
- read the validation dataset
- generate predictions
- compute MSE, MAE, and R2
- display a true-vs-predicted scatter plot
- export `regression_predictions_from_saved_model.xlsx`

## Output

The script produces:

- `regression_predictions_from_saved_model.xlsx`

with the original validation data plus two additional columns:

- `Y_pred`
- `Residual`

## Notes

- If the trained model is too large for a regular Git repository, consider using Git LFS or publishing the model archive through a release or data repository.
- If the validation data cannot be shared publicly, remove it from the public repository and provide a template file instead.

## Citation

If you use this repository in academic work, please cite the associated article and software release.

