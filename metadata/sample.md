# 2. Precision, Recall, F1-Score (Implementation)

```python
# 2. Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Classification Report (combines all metrics)
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)
```

Example output:
```
Precision: 0.77
Recall: 0.71
F1-Score: 0.74

Classification Report:
              precision    recall  f1-score   support

           0       0.81      0.85      0.83       107
           1       0.77      0.71      0.74        72

    accuracy                           0.79       179
   macro avg       0.79      0.78      0.78       179
weighted avg       0.79      0.79      0.79       179
```

# 2. Confusion Matrix

A confusion matrix is a table that is used to evaluate the performance of a classification model. It presents a summary of the predictions made by the model compared to the actual values. The matrix has four key components:

* **True Positives (TP)**: The model correctly predicted the positive class.
* **True Negatives (TN)**: The model correctly predicted the negative class.
* **False Positives (FP)**: The model incorrectly predicted the positive class (Type I error).
* **False Negatives (FN)**: The model incorrectly predicted the negative class (Type II error).

The confusion matrix helps us understand:
- Where our model is getting confused
- The types of errors our model is making
- Whether certain classes are being misclassified more often than others

## Confusion Matrix Layout

```
                   | Predicted Negative | Predicted Positive |
|-------------------|-------------------|-------------------|
| Actual Negative   |  True Negative    |  False Positive   |
| Actual Positive   |  False Negative   |  True Positive    |
```

## Implementation

```python
# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

Example output:
```
Confusion Matrix:
[[91 16]
 [21 51]]
```

The visualization of this matrix helps to clearly see where the model is performing well and where it needs improvement.

# 3. ROC Curve and AUC

ROC (Receiver Operating Characteristic) Curve and AUC (Area Under the Curve) are important evaluation metrics for classification models, particularly when dealing with binary classification problems.

## ROC Curve

The ROC curve is a graphical representation that shows the performance of a classification model at all classification thresholds. It plots:

* **True Positive Rate (Sensitivity)** on the y-axis: The proportion of actual positives correctly identified.
* **False Positive Rate (1-Specificity)** on the x-axis: The proportion of actual negatives incorrectly classified as positive.

The ROC curve illustrates the trade-off between sensitivity and specificity. By adjusting the classification threshold, we can increase one at the expense of the other.

## AUC (Area Under the Curve)

AUC provides a single scalar value that measures the overall performance of a binary classifier across all possible classification thresholds. AUC ranges from 0 to 1, where:

* **AUC = 1.0**: Perfect classifier
* **AUC = 0.5**: No better than random guessing (diagonal line on ROC plot)
* **AUC < 0.5**: Worse than random guessing

The higher the AUC, the better the model's ability to distinguish between positive and negative classes.

## Implementation

```python
# 3. ROC Curve and AUC Score
# For binary classification, we need probability estimates
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```

# Model Tuning

Model tuning is the process of optimizing the performance of a machine learning model by adjusting its hyperparameters. Hyperparameters are parameters that are not learned from the data but are set before the learning process begins.

## Hyperparameter Tuning

Hyperparameters control the behavior of the learning algorithm, and finding the optimal values can significantly improve model performance. Common methods for hyperparameter tuning include:

### Grid Search

Grid Search exhaustively searches through a predefined set of hyperparameter values. It trains a model for each combination of hyperparameters and selects the combination that produces the best performance.

### Implementation with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'solver': ['liblinear', 'saga'],  # Solver algorithms
    'max_iter': [100, 200, 300]  # Maximum iterations
}

# Initialize the model
log_reg = LogisticRegression(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Metric to optimize
    n_jobs=-1  # Use all available cores
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy of Tuned Model:", accuracy_score(y_test, y_pred))
```

## Class Imbalance

Class imbalance occurs when one class significantly outnumbers the other in a dataset, which can lead to biased models that favor the majority class. There are several strategies to address class imbalance:

### 1. Class Weights

Assigning higher weights to the minority class during training to make it "more important".

```python
from sklearn.utils import class_weight

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print("\nClass Weights:", class_weight_dict)

# Apply class weights to the model
weighted_model = LogisticRegression(class_weight=class_weight_dict, random_state=42)
weighted_model.fit(X_train, y_train)
```

### 2. Resampling Techniques

#### a. Oversampling (SMOTE)

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic examples of the minority class.

```python
from imblearn.over_sampling import SMOTE

# SMOTE for handling imbalanced data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
print("\nClass Distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Train model on resampled data
resampled_model = LogisticRegression(random_state=42)
resampled_model.fit(X_train_resampled, y_train_resampled)
```

#### b. Undersampling

Reducing the number of instances from the majority class to match the minority class.

## Overfitting and Underfitting

Overfitting and underfitting are common problems in machine learning that affect model performance:

### Overfitting

Overfitting occurs when a model learns the training data too well, including its noise and fluctuations, causing poor performance on unseen data. Signs of overfitting include:

- High training accuracy but low test accuracy
- Large gap between training and validation performance
- Complex model with many parameters

### Underfitting

Underfitting occurs when a model is too simple to capture the underlying pattern in the data. Signs of underfitting include:

- Low accuracy on both training and test data
- Small gap between training and validation performance
- Too simple model with few parameters

### Detecting Overfitting and Underfitting with Learning Curves

Learning curves show model performance as a function of training set size for both training and validation sets.

```python
from sklearn.model_selection import learning_curve

# Define a function to plot learning curves
def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy'
    )
    
    # Calculate mean and std for training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.show()

# Plot learning curve for Logistic Regression
plot_learning_curve(LogisticRegression(random_state=42), X, y)
```

### Model Complexity vs. Performance

Comparing models with different complexity levels can help identify the optimal model complexity.

```python
# Compare models with different complexity (Decision Tree with different depths)
depths = [1, 3, 5, 10, 20, None]  # None means unlimited depth
train_scores = []
test_scores = []

for depth in depths:
    depth_name = str(depth) if depth is not None else "None (unlimited)"
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    # Score on training and test sets
    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Max Depth: {depth_name}")
    print(f"  Training Score: {train_score:.4f}")
    print(f"  Test Score: {test_score:.4f}")
    print(f"  Gap: {train_score - test_score:.4f}")
    print()
```

## Cross-Validation

Cross-validation is a technique to evaluate model performance by partitioning the original data into training and testing sets multiple times. It provides a more robust assessment of model performance than a single train-test split.

### K-Fold Cross-Validation

In k-fold cross-validation, the data is divided into k subsets (or folds). The model is trained k times, each time using a different fold as the test set and the remaining folds as the training set.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Print the results
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())
```

### Benefits of Cross-Validation

1. Makes better use of the available data
2. Provides a more reliable estimate of model performance
3. Helps detect overfitting
4. Reduces the variance of the performance estimate

### Stratified K-Fold

For imbalanced datasets, stratified k-fold ensures that each fold maintains the same proportion of class labels as the original dataset.

```python
from sklearn.model_selection import StratifiedKFold

# Initialize the stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store scores
cv_scores = []

# Perform stratified k-fold cross-validation
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    # Train and evaluate model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_test_fold, y_test_fold)
    cv_scores.append(score)

print("Stratified K-Fold Cross-Validation Scores:", cv_scores)
print("Mean:", sum(cv_scores) / len(cv_scores))
```

## Summary

Model evaluation and tuning are crucial steps in the machine learning workflow. By properly evaluating models using metrics like accuracy, precision, recall, F1-score, ROC curves, and AUC, we can understand their strengths and limitations. Techniques like hyperparameter tuning, addressing class imbalance, and cross-validation help optimize models and ensure they generalize well to unseen data. By understanding the concepts of overfitting and underfitting, we can build models that strike the right balance between complexity and performance.