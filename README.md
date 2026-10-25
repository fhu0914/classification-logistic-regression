# Logistic Regression from Scores to Decisions

## Project Overview

This project implements binary logistic regression from scratch using NumPy and compares it with scikit-learn's LogisticRegression.

The Wisconsin Diagnostic Breast Cancer dataset is used with exactly two features:

- mean radius
- mean texture

The target is recoded as:

- 1 = malignant
- 0 = benign

The project also evaluates how different classification thresholds affect precision, recall, F1 score, and confusion matrix results.

## Project Structure

```text
classification-logistic-regression/
├── logistic_regression_project.py
├── README.md
└── reports/
    ├── training_loss.png
    ├── threshold_results.csv
    └── assignment_results.csv

## Requirements

- Python 3
- Git
- NumPy
- pandas
- matplotlib
- scikit-learn

## Setup and Installation

### 1. Verify Python and Git

Check that Python is installed:

```bash
python --version
```

Check that Git is installed:

```bash
git --version
```

### 2. Clone the Repository

```bash
git clone https://github.com/fhu0914/classification-logistic-regression.git
cd classification-logistic-regression
```

> Replace `<YOUR_REPO_URL>` with the actual GitHub repository URL after the repository is created.

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Project

Windows:

```bash
python logistic_regression_project.py
```

macOS:

```bash
python3 logistic_regression_project.py
```

## Expected Outputs

After the script runs successfully, the `reports` folder should contain:

- `training_loss.png`
- `threshold_results.csv`
- `assignment_results.csv`

The script also prints data checks, split sizes, model coefficients, selected threshold, and evaluation results to the terminal.

## Troubleshooting

### `python` command not found

Try:

```bash
python3 --version
```

On Windows, make sure Python was added to PATH during installation.

### Missing Python package

Run:

```bash
python -m pip install -r requirements.txt
```

### PowerShell blocks virtual environment activation

On Windows PowerShell, try:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

### Script cannot be found

Make sure the terminal is inside the repository folder before running:

```bash
python logistic_regression_project.py
```

### Expected output files are missing

After a successful run, check the `reports` folder for:

- `training_loss.png`
- `threshold_results.csv`
- `assignment_results.csv`