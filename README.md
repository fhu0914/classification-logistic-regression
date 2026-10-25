# Logistic Regression from Scores to Decisions

## Project Overview

This project implements binary logistic regression from scratch using NumPy and compares it with scikit-learn's LogisticRegression.

This project uses the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn and loaded with `sklearn.datasets.load_breast_cancer()`. The dataset contains measurements describing characteristics of cell nuclei from breast-mass samples.

Exactly two features are used:

- mean radius
- mean texture

The target is recoded as:

- 1 = malignant
- 0 = benign

The project also evaluates how different classification thresholds affect precision, recall, F1 score, and confusion matrix results.

## Project Question

Can a two-feature logistic regression model using mean radius and mean texture distinguish malignant from benign breast-mass samples, and how does changing the classification threshold affect prediction errors?

## Key Findings

- The NumPy logistic regression implementation closely matched scikit-learn's LogisticRegression.
- The selected validation threshold was 0.60, based on the highest validation F1 score.
- On the test set, the selected threshold increased precision but reduced recall and F1 compared with the default 0.50 threshold.
- The majority-class baseline achieved reasonable accuracy but failed to identify any malignant cases.
- This project is for educational purposes only and is not a clinical diagnostic tool.
- The model uses only two features from a small historical dataset, so its results should not be assumed to generalize to other patient populations or clinical settings.

For full calculations, interpretation, limitations, and the research process, see [memo.md](memo.md).

## Project Structure

```text
classification-logistic-regression/
├── .gitignore
├── README.md
├── memo.md
├── requirements.txt
├── logistic_regression_project.py
└── reports/
    ├── training_loss.png
    ├── threshold_results.csv
    └── assignment_results.csv
```    

## Requirements

- The dataset is bundled with scikit-learn and does not require a separate download.
- Python 3.10 or newer
- Git
- NumPy
- pandas
- matplotlib
- scikit-learn

## Prerequisites

Install the following software before running the project:

- Python 3.10 or newer: https://www.python.org/downloads/
- Git: https://git-scm.com/downloads

Verify the installations:

### Windows PowerShell

```powershell
python --version
git --version
```

### macOS Terminal

```bash
python3 --version
git --version
```

### How to Open a Terminal

**Windows:** Open the Start menu, search for `PowerShell`, and open Windows PowerShell.

**macOS:** Open Finder → Applications → Utilities → Terminal.

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


### 3. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

Windows:

```powershell
python -m pip install -r requirements.txt
```

macOS:

```bash
python3 -m pip install -r requirements.txt
```

### 5. Run the Project

Windows:

```powershell
python logistic_regression_project.py
```

macOS:

```bash
python3 logistic_regression_project.py
```

## Training Configuration

The NumPy logistic regression model uses:

- full-batch gradient descent
- learning rate = 0.1
- epochs = 20,000
- coefficients initialized to zero
- no regularization
- no class weights
- StandardScaler fitted on the training set only

The following thresholds are evaluated using the NumPy validation probabilities:

`0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80`

The threshold with the highest validation F1 score is selected. In this run, the selected threshold was 0.60.

## Tested Environment

This project was tested on Windows with:

- Python 3.13
- NumPy 2.4.4
- pandas 3.0.2
- Matplotlib 3.10.9
- scikit-learn 1.9.1

The full script runs in a few seconds on my Windows machine.    

## Successful Execution

A successful run should print:

- selected feature names
- missing-value counts
- class counts and split sizes
- overlap checks
- training loss
- learned NumPy coefficients
- scikit-learn coefficients
- validation probability difference
- threshold results
- final validation and test evaluation results

After the script finishes, the following files should appear in the `reports` folder:

- `training_loss.png`
- `threshold_results.csv`
- `assignment_results.csv`

Open the CSV files with Excel, a text editor, or VS Code. Open `training_loss.png` with any image viewer.

Small numerical differences may occur across Python or package versions, but the overall results and conclusions should remain similar.

## Troubleshooting

### `python` command not found

Windows PowerShell:

    py --version

macOS Terminal:

    python3 --version

If Python is installed but the command is still not recognized on Windows, make sure Python was added to PATH during installation.

### Missing Python package

Windows PowerShell:

    python -m pip install -r requirements.txt

macOS Terminal:

    python3 -m pip install -r requirements.txt

### PowerShell blocks virtual environment activation

On Windows PowerShell, run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

This changes the execution policy only for the current PowerShell session.

### Script cannot be found

Make sure the terminal is inside the repository folder.

Windows PowerShell:

    python logistic_regression_project.py

macOS Terminal:

    python3 logistic_regression_project.py

### Expected output files are missing

After a successful run, check the `reports` folder for:

- `training_loss.png`
- `threshold_results.csv`
- `assignment_results.csv`