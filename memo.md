# Logistic Regression from Scores to Decisions

## 1. Manual Logistic Regression Cases

The first part of this project demonstrated how logistic regression converts a linear score into a probability and then converts that probability into a class using a threshold. The score is calculated as \(z=\beta_0+\beta_1x\), and the sigmoid function converts \(z\) into a probability.

| Case | z | Probability | Threshold | Predicted Class |
|---|---:|---:|---:|---:|
| A | 0.6 | 0.6457 | 0.50 | 1 |
| B | 0.6 | 0.6457 | 0.70 | 0 |
| C | -2.6 | 0.0691 | 0.50 | 0 |
| D | 1.0 | 0.7311 | 0.50 | 1 |

Cases A and B show that changing the threshold changes the final classification but does not change the predicted probability. Case C shows that reversing the coefficient sign changes the direction of the relationship between the feature and the probability. In Case D, the feature coefficient is zero, so the probability is constant regardless of the input value. These coefficients were manually chosen for demonstration, unlike the coefficients learned during model training.

## 2. Data Preparation and Leakage Prevention

I used the Wisconsin Diagnostic Breast Cancer dataset from scikit-learn. Following the project requirements, I used only two features: mean radius and mean texture. The original target coding was reversed so that 1 represents malignant and 0 represents benign. The dataset contained 569 observations, including 357 benign cases and 212 malignant cases, with no missing values in the selected features.

The data were split into 341 training observations, 114 validation observations, and 114 test observations. Stratification preserved approximately the same benign/malignant proportions across all three splits. I also verified that there was no overlap among the training, validation, and test indices.

StandardScaler was fitted only on the training features. The same fitted scaler was then used to transform the training, validation, and test data. This prevents data leakage because information from the validation and test sets does not influence the preprocessing parameters.

## 3. NumPy Logistic Regression Training

I implemented binary logistic regression from scratch using NumPy. The model included an intercept and two standardized features. All three coefficients were initialized to zero. I used full-batch gradient descent with a learning rate of 0.1 for 20,000 epochs and binary cross-entropy loss.

The initial training loss was 0.6931. After training, the final loss decreased to approximately 0.2468 and then leveled off, indicating convergence. The learned coefficients were:

| Coefficient | NumPy | scikit-learn |
|---|---:|---:|
| Intercept | -0.727172 | -0.727172 |
| Mean radius | 3.656794 | 3.656794 |
| Mean texture | 1.009872 | 1.009872 |

The positive coefficients indicate that larger standardized values of mean radius or mean texture are associated with higher log-odds of malignancy, holding the other feature constant. These coefficients describe associations rather than causal effects, and they do not represent a constant change in probability.

## 4. Comparison with scikit-learn and Baseline

The NumPy coefficients were almost identical to those produced by scikit-learn LogisticRegression. The mean absolute difference between the two models' validation probabilities was approximately \(1.19 \times 10^{-10}\), providing strong evidence that the NumPy implementation was working correctly.

The DummyClassifier baseline learned only the training class proportions. At a threshold of 0.50, it predicted every validation observation as benign. Its validation accuracy was approximately 0.623, but malignant recall and F1 were both 0. This demonstrates why accuracy alone can be misleading: the baseline classified the majority class reasonably often but failed to identify any malignant cases.

## 5. Threshold Selection and Final Evaluation

Using NumPy validation probabilities, I evaluated thresholds from 0.20 through 0.80. The highest validation F1 score was 0.8312 at a threshold of 0.60, so 0.60 was selected and frozen before test evaluation.

On validation data, changing the threshold from 0.50 to 0.60 reduced false positives from 6 to 2 and increased precision from 0.8500 to 0.9412. However, false negatives increased from 9 to 11 and recall decreased from 0.7907 to 0.7442. The resulting F1 increased from 0.8193 to 0.8312.

On the test set, the selected threshold did not improve F1. At 0.50, the NumPy model had 4 false positives, 11 false negatives, precision of 0.8857, recall of 0.7381, and F1 of 0.8052. At 0.60, false positives decreased to 3, but false negatives increased to 14. Precision increased to 0.9032, while recall decreased to 0.6667 and F1 decreased to 0.7671. I did not change the threshold after observing the test results. Log loss remained 0.2557 for both thresholds because threshold changes affect class decisions, not the underlying predicted probabilities.

## 6. Limitations and Clinical Caution

This model is an educational exercise and should not be used for clinical decision-making. First, only two features were included, so other potentially important diagnostic information was excluded. Second, performance on this dataset may not generalize to all real-world patient populations. Third, F1 was used because it was the required threshold-selection rule for this assignment, but it does not directly represent the real clinical consequences of false negatives and false positives. In a real cancer-screening setting, missing a malignant case may carry a very different cost from generating a false alarm.

## Research Process

During implementation, I first verified the selected features, missing values, target coding, split sizes, class proportions, and absence of overlap between splits. I then confirmed that the standardized training features had means near zero and standard deviations near one. Before full training, I tested the sigmoid function at \(z=0\) and confirmed that it returned 0.5.

I also encountered an environment issue because scikit-learn was not initially installed. After installing the required package, I continued the implementation. To validate the NumPy gradient descent code, I compared its coefficients and validation probabilities with scikit-learn. Their extremely small differences provided evidence that the implementation was correct.