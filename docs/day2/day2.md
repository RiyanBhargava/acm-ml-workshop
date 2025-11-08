# ML Workshop Day 2: Model Training and Analysis
## Complete Documentation Guide

---

## 📋 Table of Contents
1. Overview
2. Dataset Information
3. Overfitting and Underfitting
4. Train-Test Split
5. Types of ML Problems
6. Regression Models
7. Classification Models
8. Evaluation Metrics
9. Model Comparison
10. Key Insights

---

## 🎯 Overview

This workshop introduces fundamental machine learning concepts with practical implementation. You'll learn how to:
- Split data for training and testing
- Implement multiple regression and classification models
- Evaluate model performance
- Compare and select the best model

**Dataset**: Real estate pricing data 

---

## 📊 Dataset Information

### Original Dataset
- **Total Records**: 10,835 properties
- **Features**: 246 (after preprocessing)
- **Target Variable**: `price` (continuous numerical value) - that we want to predict

### Feature Preparation
```python
# Separating features and target
X = df[feature_cols]  # All columns except 'price'
y = df['price']       # Target variable

# Dataset shape
Features: (10835, 244)
Target: (10835,)
```

---

## Overfitting and Underfitting
When building machine learning models, the goal is to **capture the true underlying patterns in data** so the model can **generalize** to new, unseen examples.  

However, models can sometimes go wrong in two common ways:

1. **Underfitting**
2. **Overfitting** 

![Underfit,Goodfit and Overfit curves](../assets/image13.png)

Striking the **right balance** between underfitting and overfitting is key to building robust machine learning models.

### Overfitting
Overfitting happens when a model learns too much from the training data, including details that don’t matter (like noise or outliers).

**Example :**
- Imagine fitting a very complicated curve to a set of points. The curve will go through every point, but it won’t represent the actual pattern.
- As a result, the model works great on training data but fails when tested on new data.

![An example of overfit curve](../assets/image14.png)

**Reasons for Overfitting:**

1. High variance and low bias.
2. The model is too complex.
3. The size of the training data.

### Underfitting

Underfitting is the opposite of overfitting. It happens when a model is too simple to capture what’s going on in the data.

**Example:**
- Imagine drawing a straight line to fit points that actually follow a curve. The line misses most of the pattern.
- In this case, the model doesn’t work well on either the training or testing data.

![An example of underfit curve](../assets/image15.png)

**Reasons for Underfitting:**

1. The model is too simple, So it may be not capable to represent the complexities in the data.
2. The input features which is used to train the model is not the adequate representations of underlying factors influencing the target variable.
3. The size of the training dataset used is not enough.
4. Features are not scaled.

## 🔀 Train-Test Split

### What is Train-Test Split?

Train-test split divides your dataset into two parts:

![Visual Representation of Train-Test Split](../assets/image1.png)


**Training Set (80%)**: Used to teach the model
- Model learns patterns from this data
- Used for fitting/training algorithms

**Testing Set (20%)**: Used to evaluate the model
- Model has never seen this data
- Tests how well model generalizes to new data

### Implementation
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)
```

### Why Split Data?
- **Prevents Overfitting**: Model doesn't memorize training data
- **Tests Generalization**: Evaluates performance on unseen data
- **Realistic Performance**: Simulates real-world predictions

---
## Types of Machine Learning Problems
In supervised learning, problems are usually divided into two types : 
- Regression Problem
- Classification Problem

### Regression Problem
- **Goal :** To predict a continuous numeric value.
- Regression models try to find relationships between input variables (features) and a continuous output.

- Examples:

    - Predicting house prices 🏠
    - Estimating temperature 🌡️
    - Forecasting stock prices 📈  


- Common Algorithms:

    1. Linear Regression    
    2. Decision Tree Regressor
    3. Random Forest Regressor
    4. Support Vector Regressor (SVR)

- Evaluation Metrics:

    1. Mean Squared Error (MSE)
    2. Root Mean Squared Error (RMSE)
    3. Mean Absolute Error (MAE)
    4. R² Score

### Classification Problem
- **Goal :** To predict a discrete label or category.
- Classification models learn to separate data into different classes.

- Examples:

    - Email spam detection ✉️
    - Disease diagnosis (positive/negative) 🧬
    - Image recognition (cat vs. dog) 🐱🐶 


- Common Algorithms:

    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    4. Support Vector Machine (SVM)
    5. k-Nearest Neighbors (KNN)

- Evaluation Metrics:

    1. Accuracy
    2. Precision & Recall
    3. F1 Score
    4. Confusion Matrix

---

## 📈 Regression Models

Regression predicts **continuous numerical values** (e.g., house prices, temperature, sales).

--- 

### 1. Linear Regression

**How it works**: Finds the best straight line through your data points.

**Mathematical Formula**: 
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Where:
ŷ = predicted value
β₀ = intercept (bias)
β₁, β₂, ..., βₙ = coefficients (weights)
x₁, x₂, ..., xₙ = feature values
```

Simple form: ```y = mx + b```
- Simple and interpretable
- Assumes linear relationship between features and target

![Linear Regression Diagram](../assets/image2.png)

**Strengths**:
- Fast to train
- Easy to interpret
- Works well with linear relationships

**Weaknesses**:
- Cannot capture complex non-linear patterns
- Sensitive to outliers - basically those values that are much out of range when compared to normal values

**Use cases :**
- Predicting house prices based on area, location, etc.  
- Estimating sales revenue from advertising spend.  
- Forecasting demand or performance metrics. 
---

### 2. Decision Tree Regressor

**How it works**: Creates a tree of yes/no questions to make predictions.

**Example**:
```
Is size > 2000 sq ft?
  ├─ Yes → Is location = downtown?
  │         ├─ Yes → Predict $500k
  │         └─ No → Predict $350k
  └─ No → Predict $250k
```


![Decision Tree Diagram](../assets/image3.png)

**Mathematical Formula :**
```
Prediction at leaf node = (1/n) Σᵢ₌₁ⁿ yᵢ

Where:
n = number of samples in the leaf
yᵢ = actual values in the leaf
(Takes the mean of training samples that reach that leaf)

Split criterion (MSE):
MSE = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷ)²
```
**Strengths**:
- Handles non-linear relationships
- Easy to visualize and understand
- No feature scaling needed

**Weaknesses**:
- Can overfit easily
- Sensitive to small data changes
- May create overly complex trees

**Use cases :**
- Predicting sales based on season, location, and marketing.  
- Modeling complex, non-linear data patterns.  
---

### 3. Random Forest Regressor

**How it works**: Creates many decision trees and averages their predictions.

**Think of it as**: A committee of experts voting on the answer
- Each tree sees slightly different data
- Final prediction = average of all trees
- Reduces overfitting compared to single tree

![Random Forest Regressor Diagram](../assets/image4.png)

**Mathematical Formula :**
```
ŷ = (1/T) Σₜ₌₁ᵀ hₜ(x)

Where:
T = number of trees in the forest
hₜ(x) = prediction from tree t
ŷ = final prediction (average of all trees)
```
**Strengths**:
- More accurate than single decision tree
- Handles complex relationships
- Reduces overfitting
- Shows feature importance

**Weaknesses**:
- Slower to train
- Less interpretable
- Requires more memory

**Use Cases :**
- Predicting house prices, insurance claim amounts.  
- Forecasting demand or energy consumption. 
---

### 4. Support Vector Regressor (SVR)

**How it works**: 
- Fits a line or curve that predicts most of the data within a “tube” of tolerance (𝜖).
- Focuses only on points that lie on or outside the tube (support vectors).
- Can handle nonlinear patterns using kernel functions (like RBF).

**Key Concept**: 
- Uses kernel tricks for non-linear patterns
- Focuses on data points that define boundaries
- RBF kernel used in your implementation

**Mathematical Formula :**
```
Minimize: (1/2)||w||² + C Σᵢ₌₁ⁿ (ξᵢ + ξᵢ*)

Subject to:
|yᵢ - (w·xᵢ + b)| ≤ ε + ξᵢ

Where:
w = weight vector
ε = epsilon (tube width)
C = penalty parameter
ξᵢ = slack variables
```

![Support Vector Regressor Diagram](../assets/image5.png)

**Strengths**:
- Effective in high-dimensional spaces
- Memory efficient
- Robust to outliers

**Weaknesses**:
- Slower on large datasets
- Needs feature scaling
- Difficult to interpret

**Use Cases :**
- Predicting stock prices or exchange rates.  
- Estimating real estate prices where outliers exist. 
---

### 5. K-Nearest Neighbors (KNN) Regressor

**How it works**: Predicts based on the K closest training examples.

**Example** (K=5):
- Find 5 nearest houses to your property
- Average their prices
- That's your prediction

**Mathematical Formula :**
```
ŷ = (1/K) Σᵢ₌₁ᴷ yᵢ

Where:
K = number of nearest neighbors
yᵢ = value of i-th nearest neighbor

Distance (Euclidean):
d(x, xᵢ) = √(Σⱼ₌₁ⁿ (xⱼ - xᵢⱼ)²)
```

![KNN Regressor Diagram](../assets/image6.png)

**Strengths**:
- Simple to understand
- No training phase (lazy learning)
- Naturally handles non-linear patterns

**Weaknesses**:
- Slow predictions on large datasets
- Needs feature scaling
- Sensitive to irrelevant features

**Use Cases :**
- Estimating house rent based on nearby similar properties.  
- Predicting temperature using data from nearby weather stations.

---

## 🎯 Classification Models

Classification predicts **categories/classes** (e.g., spam/not spam, disease/healthy, high/medium/low price).

### 1. Logistic Regression

**How it works**: Despite the name, it's for classification! Predicts probability of belonging to a class.

**Example**: Predicting if house is "expensive" or "affordable"
```
Probability = 1 / (1 + e^(-score))
If probability > 0.5 → Expensive
If probability ≤ 0.5 → Affordable
```

**Mathematical Formula :**
```
P(y=1|x) = 1 / (1 + e^(-z))

Where:
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
P(y=1|x) = probability of class 1
e = Euler's number (≈2.718)

Decision: If P(y=1|x) > 0.5 → Class 1
          If P(y=1|x) ≤ 0.5 → Class 0
```

![Logistic Regression Diagram](../assets/image7.png)

**Strengths**:
- Fast and efficient
- Provides probability scores
- Easy to interpret

**Weaknesses**:
- Assumes linear decision boundary
- Not effective for complex relationships

**When to use**: Binary classification with linearly separable data

---

### 2. Decision Tree Classifier

**How it works**: Same tree structure as regression, but predicts categories.

**Example**:
Is size > 2000 sq ft?
  ├─ Yes → Is location = downtown?
  │         ├─ Yes → Class: Luxury
  │         └─ No → Class: Standard
  └─ No → Class: Budget

**Mathematical Formula :**
```
Gini Impurity = 1 - Σᵢ₌₁ᶜ pᵢ²

Where:
c = number of classes
pᵢ = proportion of class i in node

Entropy (alternative):
H = -Σᵢ₌₁ᶜ pᵢ log₂(pᵢ)

(Tree splits to minimize impurity)
```

![Decision Tree Classifier](../assets/image8.png)

**Strengths**:
- Handles non-linear boundaries
- Interpretable
- Works with categorical data

**Weaknesses**:
- Overfits easily
- Unstable with small data changes

**When to use**: When you need interpretability and have categorical data

---

### 3. Random Forest Classifier

**How it works**: Ensemble of decision trees voting on the class.

**Voting Example** (5 trees):
- Tree 1: Luxury
- Tree 2: Standard
- Tree 3: Luxury
- Tree 4: Luxury
- Tree 5: Standard
- **Final Prediction**: Luxury (majority vote: 3/5)

**Mathematical Formula :**
```
ŷ = mode{h₁(x), h₂(x), ..., hₜ(x)}

Where:
T = number of trees
hₜ(x) = prediction from tree t
mode = most frequent class (majority vote)

For probabilities:
P(class=c|x) = (1/T) Σₜ₌₁ᵀ I(hₜ(x) = c)
```

![Random Forest Classifier](../assets/image9.png)

**Strengths**:
- High accuracy
- Reduces overfitting
- Shows feature importance
- Handles imbalanced data well

**Weaknesses**:
- Slower than single tree
- Less interpretable
- More memory intensive

**When to use**: When accuracy is priority and you have sufficient data

---

### 4. Support Vector Machine (SVM) Classifier

**How it works**: Finds the best boundary (hyperplane) that separates classes with maximum margin.

**Mathematical Formula :**
```
Minimize: (1/2)||w||² + C Σᵢ₌₁ⁿ ξᵢ

Subject to:
yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ

Where:
w = weight vector (perpendicular to hyperplane)
b = bias
C = penalty parameter
ξᵢ = slack variables
yᵢ ∈ {-1, +1} = class labels
```

![SVM Classifier](../assets/image10.png)

**Strengths**:
- Effective in high dimensions
- Works well with clear margins
- Memory efficient

**Weaknesses**:
- Slow on large datasets
- Needs parameter tuning
- Requires feature scaling

**When to use**: High-dimensional data with clear separation

---

### 5. K-Nearest Neighbors (KNN) Classifier

**How it works**: Assigns class based on K nearest neighbors' majority vote.

**Example** (K=5):
- Find 5 nearest houses
- 3 are "Luxury", 2 are "Standard"
- Predict: "Luxury" (majority)

**Mathematical Formula :**
```
ŷ = mode{y₁, y₂, ..., yₖ}

Where:
K = number of nearest neighbors
yᵢ = class of i-th nearest neighbor
mode = most frequent class

Distance (Euclidean):
d(x, xᵢ) = √(Σⱼ₌₁ⁿ (xⱼ - xᵢⱼ)²)
```

![KNN Classifier](../assets/image11.png)

**Strengths**:
- Simple and intuitive
- No training needed
- Naturally handles multi-class

**Weaknesses**:
- Slow for large datasets
- Sensitive to feature scaling
- Curse of dimensionality

**When to use**: Small to medium datasets with good feature engineering

---

### 6. Naive Bayes Classifier

**How it works**: Uses Bayes' Theorem assuming features are independent.

**Mathematical Formula**: 

**Bayes' Theorem:**
```P(A | B) = [ P(B | A) * P(A) ] / P(B)```

**For Naïve Bayes Classification:**
```
P(C | X) = [ P(X | C) * P(C) ] / P(X)

Where:
- P(C | X) → Posterior probability of class C given predictor X  
- P(X | C) → Likelihood of predictor given class  
- P(C) → Prior probability of class  
- P(X) → Probability of predictor (same for all classes)
```
**Naïve (Independence) Assumption:**
```P(X | C) = P(x₁, x₂, ..., xₙ | C) = Π P(xᵢ | C)```

**Final Formula:**
```P(C | X) ∝ P(C) * Π P(xᵢ | C)```

![Naive Bayes Classifier](../assets/image12.png)

**Example**: Weather Prediction (Naïve Bayes)

- We want to predict whether someone will **play tennis (Yes/No)** based on the **weather conditions**.

- Suppose we have features:
   - Outlook = Sunny, Overcast, or Rain  
   - Temperature = Hot, Mild, or Cool  
   - Humidity = High or Normal  

- We want to find:  
```P(Play = Yes | Outlook = Sunny, Humidity = High)```

- Using Naïve Bayes:
```
P(Play = Yes | Outlook = Sunny, Humidity = High) ∝ 
P(Play = Yes) * P(Outlook = Sunny | Play = Yes) * P(Humidity = High | Play = Yes)
```

- Each probability is estimated from training data (frequency counts).  
- The class (Yes or No) with the higher probability becomes the prediction.


**Strengths**:
- Very fast
- Works well with text data
- Needs little training data
- Handles multi-class naturally

**Weaknesses**:
- Assumes feature independence (often unrealistic)
- Cannot learn feature interactions

**When to use**: Text classification, spam detection, sentiment analysis

---

## 📊 Evaluation Metrics

### Regression Metrics

#### 1. Mean Squared Error (MSE)
**Formula**: Average of squared differences between predictions and actual values  
```MSE = (1/n) * Σ (yᵢ - ŷᵢ)²```

**Interpretation**:
- Lower is better
- Heavily penalizes large errors
- Units are squared (e.g., dollars²)

**Example**: 
- Actual: $300k, Predicted: $310k → Error²: (10k)² = 100M
- Actual: $300k, Predicted: $320k → Error²: (20k)² = 400M
- MSE = (100M + 400M) / 2 = 250M
---

#### 2. Root Mean Squared Error (RMSE)
**Formula**: Square root of MSE  
```RMSE = √[ (1/n) * Σ (yᵢ - ŷᵢ)² ]```

**Interpretation**:
- Lower is better
- Same units as target (dollars, not dollars²)
- More interpretable than MSE

**Example**:
- From above, MSE = 250M  
- RMSE = √250M ≈ 15.8k
---

#### 3. Mean Absolute Error (MAE)
**Formula**: Average of absolute differences  
```MAE = (1/n) * Σ |yᵢ - ŷᵢ|```

**Interpretation**:
- Lower is better
- Less sensitive to outliers than RMSE
- Direct average error

**Example:**  
- Actual: $300k, Predicted: $310k → |Error| = 10k  
- Actual: $300k, Predicted: $320k → |Error| = 20k  
- MAE = (10k + 20k) / 2 = 15k
---

#### 4. R² Score (R-Squared)
**Formula**: 1 - (Sum of Squared Residuals / Total Sum of Squares)  
```R² = 1 - [ Σ (yᵢ - ŷᵢ)² / Σ (yᵢ - ȳ)² ]```

**Interpretation**:
- Range: -∞ to 1.0
- 1.0 = Perfect predictions
- 0.0 = Model no better than predicting mean
- < 0 = Model worse than predicting mean

**Example:**  
- Total variance (Σ(yᵢ - ȳ)²) = 1000M  
- Residual variance (Σ(yᵢ - ŷᵢ)²) = 250M  
- R² = 1 - (250 / 1000) = 0.75 → Model explains 75% of variance

---

### Classification Metrics

Let us assume we have:

| Actual        | Predicted     |
|---------------|---------------|
| Positive (1)  | Positive (1)  |
| Negative (0)  | Positive (1)  |
| Positive (1)  | Negative (0)  |
| Negative (0)  | Negative (0)  |


So:  
TP = 1, TN = 1, FP = 1, FN = 1

---
#### 1. Accuracy
**Formula**: (Correct Predictions) / (Total Predictions)  
```Accuracy = (TP + TN) / (TP + TN + FP + FN)```

**Example**:
Accuracy = (1 + 1) / (1 + 1 + 1 + 1) = 0.5 → **50% accuracy**

**Limitation**: Misleading with imbalanced classes

---

#### 2. Confusion Matrix
Compares predictions vs actual:

```
                Predicted
              No    Yes
Actual  No   [TN]  [FP]
        Yes  [FN]  [TP]
```

- **TP**: True Positives (correctly predicted Yes)
- **TN**: True Negatives (correctly predicted No)
- **FP**: False Positives (predicted Yes, actually No)
- **FN**: False Negatives (predicted No, actually Yes)

---

#### 3. Precision
**Formula**:   
```TP / (TP + FP)```

**Meaning**: "Of all positive predictions, how many were correct?"

**Example**:
 = 1 / (1 + 1) = 0.5 → **50% of predicted positives are correct**

---

#### 4. Recall (Sensitivity)
**Formula**:   
```TP / (TP + FN)```

**Meaning**: "Of all actual positives, how many did we catch?"

**Example :**
= 1 / (1 + 1) = 0.5 → **50% of actual positives identified**

---

#### 5. F1-Score
**Formula**:   
```F1 = 2 * (Precision * Recall) / (Precision + Recall)```

**Meaning**: Harmonic mean of precision and recall

**Example :**
F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5

**When to use**: Balances precision and recall, especially with imbalanced data

---

## 🏆 Model Comparison (Regression Results)

### Regression Performance Ranking

| Rank | Model | R² Score | RMSE | MAE |
|------|-------|----------|------|-----|
| 🥇 1 | Linear Regression | 0.7904 | 30.79 | 9.95 |
| 🥈 2 | Random Forest | 0.7375 | 34.45 | 1.77 |
| 🥉 3 | KNN | 0.6585 | 39.29 | 1.91 |
| 4 | Decision Tree | 0.6268 | 41.08 | 3.14 |
| 5 | SVR | 0.5188 | 46.64 | 3.88 |

### Key Observations

**Linear Regression wins because**:  

- ✅ Highest R² score (79.04%)  
- ✅ Lowest RMSE (best average error)  
- ✅ Fast training and prediction  
- ✅ Easy to interpret  

**Interesting finding**: Despite lower MAE, Random Forest has better overall performance metrics than simpler models.

---

## 🏆 Model Comparison (Classification Results)

### Classification Performance Ranking

| Model                | Accuracy | Precision | Recall | F1 Score |
|-----------------------|-----------|------------|---------|-----------|
| Logistic Regression   | 0.9231    | 0.9479     | 0.9231  | 0.9271    |
| Decision Tree         | 1.0000    | 1.0000     | 1.0000  | 1.0000    |
| Random Forest         | 1.0000    | 1.0000     | 1.0000  | 1.0000    |
| SVM                   | 0.6667    | 0.5041     | 0.6667  | 0.5698    |
| KNN                   | 0.6410    | 0.6282     | 0.6410  | 0.6197    |
| Naive Bayes           | 0.9487    | 0.9615     | 0.9487  | 0.9508    |

### Confusion Matrix for all models 

![Confusion Matrix for all Models](../assets/image16.png)

### Key Observations

🏆 **BEST CLASSIFICATION MODELS**

| Model          | Accuracy | Precision | Recall | F1 Score |
|----------------|-----------|------------|---------|-----------|
| Decision Tree  | 1.0       | 1.0        | 1.0     | 1.0       |
| Random Forest  | 1.0       | 1.0        | 1.0     | 1.0       |

Decision Tree & Random Forest win because:

✅ Perfect test accuracy on this dataset
✅ Can capture complex, non-linear relationships
✅ Handle both categorical and numerical features naturally
✅ Robust and flexible for small datasets

---

### General ML Best Practices

1. **Always split your data**
   - Train-test split prevents overfitting
   - Use cross-validation for robust evaluation

2. **Try multiple models**
   - Different models work better for different data
   - No "one size fits all" solution

3. **Understand your metrics**
   - R² for overall model fit
   - RMSE for average prediction error
   - MAE for median error magnitude

4. **Consider the business context**
   - Is $31k error acceptable for your use case?
   - Sometimes a simple, interpretable model is better than a complex one
---

## 📚 Summary

**You've learned**:

- ✅ Train-test split methodology
- ✅ 5 regression algorithms
- ✅ 6 classification algorithms 
- ✅ Multiple evaluation metrics
- ✅ Model comparison techniques

### 📥 Download Material

- 📓 Download Notebook for Regression Problem:  
  [ML_Worshop_Day2](../files/day2/ML_Workshop_Day_2.ipynb)

- 📊 Download Preprocessed Dataset for Regression Problem(CSV):  
  [preprocessed_dataset.csv](../files/day2/houseprice_preprocessed_data.csv)

- 📓 Download Notebook for Regression Problem:  
  [ML_Worshop_Day2](../files/day2/ML_Workshop_Day2_Classification.ipynb)

- 📊 Download Preprocessed Dataset for Classification Problem(CSV):  
  [preprocessed_dataset.csv](../files/day2/drugclassification_preprocessed_data.csv)
Run each cell step by step to see data cleaning and feature engineering in action!

---

**Remember**: 
The best model isn't always the most complex one.   
Choose based on:
- Performance on test data
- Interpretability needs
- Computational resources
- Business requirements

---

*Created for ML Workshop Day 2 | Happy Learning! 🎓*