# Logistic Regression from Scores to Decisions

## 1. Manual Logistic Regression Cases

Logistic regression converts a linear score into a probability using the sigmoid function and then applies a threshold to assign a class.

| Case | z | Probability | Threshold | Predicted Class |
|---|---:|---:|---:|---:|
| A | 0.6 | 0.6457 | 0.50 | 1 |
| B | 0.6 | 0.6457 | 0.70 | 0 |
| C | -2.6 | 0.0691 | 0.50 | 0 |
| D | 1.0 | 0.7311 | 0.50 | 1 |

For Case A, $z=-1.0+0.8(2.0)=0.6$. Applying the sigmoid gives $p=1/(1+\exp(-0.6))=0.6457$. Because 0.6457 is greater than the 0.50 threshold, the predicted class is 1.

Cases A and B show that changing the threshold changes the class decision but not the probability. Case C shows that reversing the coefficient changes the direction of the relationship. In Case D, the coefficient is zero, so the input has no effect and the probability remains constant. These coefficients were manually specified, unlike the coefficients learned during training.

## 2. Data Preparation and Leakage Prevention

I used the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn through `load_breast_cancer()`. According to `data.DESCR`, the measurements describe characteristics of cell nuclei from breast-mass samples. I used only `mean radius` and `mean texture`. The target was recoded so that 1 = malignant and 0 = benign.

The dataset contained 569 observations: 357 benign and 212 malignant, with no missing values in the selected features. The split produced 341 training, 114 validation, and 114 test observations. Stratification preserved similar class proportions, the three partitions had no overlap, and together they covered all rows.

`StandardScaler` was fitted only on the training features and then reused for validation and test data. This prevents leakage because validation and test information does not influence preprocessing.

## 3. NumPy Logistic Regression Training

I implemented logistic regression from scratch using NumPy with an intercept, two standardized features, full-batch gradient descent, a learning rate of 0.1, and 20,000 epochs. All coefficients were initialized to zero.

Training loss decreased from 0.6931 to approximately 0.2468 and then leveled off, indicating convergence.

| Coefficient | NumPy | scikit-learn |
|---|---:|---:|
| Intercept | -0.727172 | -0.727172 |
| Mean radius | 3.656794 | 3.656794 |
| Mean texture | 1.009872 | 1.009872 |

Minimizing binary cross-entropy is equivalent to minimizing negative log-likelihood, so training seeks coefficients that make the observed class labels more likely. Because mean radius was standardized using the training data, its coefficient of 3.6568 means that a one-training-standard-deviation increase in mean radius is associated with a 3.6568 increase in the log-odds of malignancy, holding mean texture fixed. This is not a constant probability change and does not imply causation.

## 4. Comparison with scikit-learn and Baseline

The NumPy coefficients were almost identical to the scikit-learn coefficients. The mean absolute difference between their validation probabilities was approximately $1.19 \times 10^{-10}$, supporting the correctness of the NumPy implementation.

The `DummyClassifier` baseline predicted every validation case as benign at threshold 0.50. Its accuracy was about 0.623, but recall and F1 for malignant cases were both 0. Its confusion counts were TN = 71, FP = 0, FN = 43, and TP = 0. In plain language, it correctly labeled all 71 benign cases but missed all 43 malignant cases. Because it predicted no positive cases, `zero_division=0` reports precision as 0. This shows why accuracy alone can be misleading.

## 5. Threshold Selection and Final Evaluation

Using NumPy validation probabilities, I evaluated thresholds from 0.20 to 0.80. The highest validation F1 was 0.8312 at threshold 0.60, so 0.60 was selected and frozen before test evaluation.

On validation data, changing the threshold from 0.50 to 0.60 reduced false positives from 6 to 2 and increased precision from 0.8500 to 0.9412. However, false negatives increased from 9 to 11, recall decreased from 0.7907 to 0.7442, and F1 increased from 0.8193 to 0.8312.

On the test set, the selected threshold did not improve F1. At 0.50, precision was 0.8857, recall was 0.7381, and F1 was 0.8052. At 0.60, precision increased to 0.9032, but recall fell to 0.6667 and F1 fell to 0.7671. I did not change the threshold after seeing the test results. Log loss remained 0.2557 because threshold changes affect class labels, not the underlying probabilities.

## 6. Limitations and Clinical Caution

This model is an educational exercise, not a clinical decision tool. It uses only two features, so important diagnostic information may be excluded, and results from this historical dataset may not generalize to other patient populations. In addition, F1 was required for threshold selection but does not represent the real clinical costs of different errors. Missing a malignant case may have much more serious consequences than generating a false positive.

## Research Process

I first verified feature selection, target coding, missing values, split sizes, class counts, class proportions, row coverage, and absence of overlap. I also confirmed that standardized training features had means near zero and standard deviations near one.

Before full training, I tested the sigmoid function at $z=0$ and confirmed that it returned 0.5. I encountered an environment issue because scikit-learn was not initially installed, so I installed the required package and reran the project.

To validate the NumPy implementation, I compared its learned coefficients and validation probabilities with the scikit-learn reference model. Their extremely small differences provided evidence that the implementation and preprocessing pipeline were working correctly.