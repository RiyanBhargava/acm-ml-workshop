# Day 2: Model Building and Evaluation

## Welcome Back!

Yesterday, we learned the crucial art of data cleaning and feature engineering. Today, we're going to build on that foundation and create actual machine learning models that can predict house prices! This is where the magic happens - where data transforms into intelligent predictions.

## Understanding Machine Learning Models

Before we dive in, let's demystify what a machine learning model actually is.

### What is a Machine Learning Model?

Think of a machine learning model as a mathematical recipe that learns patterns from data. It's like teaching a child to recognize animals:

1. **Show examples**: "This is a cat, this is a dog, this is a bird"
2. **Child learns patterns**: "Cats have pointy ears, dogs bark, birds have wings"
3. **Test understanding**: Show a new picture - can the child identify it?

Machine learning works similarly:
1. **Training**: Feed the model data (features + prices)
2. **Learning**: Model finds mathematical relationships between features and prices
3. **Prediction**: Given new features, model predicts the price

---

## The Training-Testing Split

### Why Split the Data?

Imagine studying for an exam using practice questions, then taking the same exam with the exact same questions. Your score wouldn't truly reflect your understanding - you might have just memorized answers!

**The Principle**: Never test a model on data it has seen during training.

**The Solution**: Split data into two parts:
- **Training Set (80%)**: Used to teach the model patterns
- **Testing Set (20%)**: Used to evaluate how well it learned

### How Does This Work?

Let's say you have 1,000 houses:
- Train on 800 houses: Model learns "In Location A, 2 BHK with 1000 sq ft typically costs ₹50 lakhs"
- Test on 200 new houses: Does the model predict correctly for houses it hasn't seen?

**Key Insight**: Good performance on training data is easy. Good performance on test data means the model truly understands patterns, not just memorized examples.

---

## Regression: The Foundation

### What is Regression?

Regression is a type of machine learning used when you want to predict a **continuous numerical value** (like price, temperature, or salary).

**Contrast with Classification**:
- Regression: "This house costs ₹65.5 lakhs" (continuous number)
- Classification: "This email is spam/not spam" (category)

### The Regression Goal

Find a mathematical function that best describes the relationship between:
- **Input (X)**: Square feet, bedrooms, bathrooms, location
- **Output (y)**: Price

**In mathematical terms**: y = f(X)

We want to find the function f that makes the most accurate predictions.

---

## Model 1: Linear Regression

### The Concept

Linear Regression assumes a straight-line relationship between features and the target.

**Simple Example**: 
- Hypothesis: Price = (Price per sq ft × Square feet) + Base price
- If price per sq ft = ₹5,000 and base price = ₹10 lakhs
- For 1,000 sq ft: Price = (5,000 × 1,000) + 10,00,000 = ₹60 lakhs

**In Real Life**: The relationship involves multiple features:
Price = (coefficient₁ × sqft) + (coefficient₂ × bedrooms) + (coefficient₃ × bathrooms) + ... + intercept

### How Linear Regression Learns

1. **Initialize**: Start with random coefficients
2. **Predict**: Calculate predicted prices using current coefficients
3. **Calculate Error**: Compare predictions with actual prices
4. **Adjust**: Modify coefficients to reduce error
5. **Repeat**: Continue until error is minimized

**Analogy**: It's like adjusting the temperature knobs on a shower until the water is perfect - small adjustments based on feedback.

### Strengths of Linear Regression

- ✅ Simple and interpretable
- ✅ Fast to train
- ✅ Works well when relationships are approximately linear
- ✅ Requires less data than complex models

### Limitations

- ❌ Assumes linear relationships (reality is often non-linear)
- ❌ Sensitive to outliers (though we removed most!)
- ❌ May underfit complex patterns

---

## Model 2: Decision Tree Regression

### Thinking in Trees

Decision Trees make predictions by asking a series of yes/no questions.

**Real-Life Example**: Deciding if a house is expensive:
1. Is it in a prime location? 
   - Yes → Go to question 2
   - No → Predict "Budget range"
2. Is it larger than 2000 sq ft?
   - Yes → Predict "Premium range"
   - No → Predict "Mid range"

### How Decision Trees Work for Regression

Instead of categories, trees predict numbers by splitting data into groups.

**Example Process**:
1. **First Split**: "Is square footage > 1500?"
   - Left branch (≤1500): Average price = ₹40 lakhs
   - Right branch (>1500): Average price = ₹75 lakhs

2. **Second Split on Left Branch**: "Are bedrooms ≤ 2?"
   - Yes: Predict ₹35 lakhs
   - No: Predict ₹50 lakhs

3. **Continue splitting** until groups are homogeneous or stopping criteria are met

### Key Parameters

**Criterion**: How to measure the quality of a split
- **Squared Error**: Minimize variance in each group (how spread out the prices are)
- **Friedman MSE**: Similar but with adjustments for better performance

**Splitter**: How to choose the split point
- **Best**: Always choose the split that most reduces error (thorough but slower)
- **Random**: Choose from random splits (faster but potentially less optimal)

### Strengths of Decision Trees

- ✅ Capture non-linear relationships naturally
- ✅ Easy to understand and visualize
- ✅ Handle feature interactions automatically
- ✅ No need for feature scaling

### Limitations

- ❌ Prone to overfitting (memorizing training data)
- ❌ Small changes in data can drastically change the tree
- ❌ Can create overly complex models

---

## Model 3: Random Forest Regression

### The Wisdom of Crowds

Imagine asking one person for house price advice vs. asking 100 real estate experts and averaging their opinions. Which would you trust more?

Random Forest uses this "wisdom of crowds" principle by creating many decision trees and averaging their predictions.

### How Random Forest Works

**The Process**:
1. **Create multiple datasets**: Randomly sample data with replacement (bootstrap sampling)
2. **Build many trees**: Train a decision tree on each sampled dataset
3. **Random feature selection**: Each tree only considers a random subset of features at each split
4. **Aggregate predictions**: Average all tree predictions for final output

**Example**: 
- Tree 1 predicts: ₹60 lakhs
- Tree 2 predicts: ₹58 lakhs
- Tree 3 predicts: ₹62 lakhs
- ...
- Tree 100 predicts: ₹59 lakhs
- **Final prediction**: Average = ₹60 lakhs

### Why is This Better?

**Analogy**: If one doctor misdiagnoses, it's a problem. But if 100 doctors independently examine a patient and vote on the diagnosis, the collective opinion is usually more reliable.

**Benefits of Randomness**:
- Different trees make different mistakes
- Errors cancel out when averaged
- More stable and robust predictions

### Key Parameters

**n_estimators**: Number of trees in the forest
- More trees = Better performance (but slower)
- Typical values: 100-500

**max_depth**: Maximum depth of each tree
- Controls how complex each individual tree can be
- Prevents overfitting

**max_features**: Number of features to consider at each split
- 'auto' or 'sqrt': √(total features)
- 'log2': log₂(total features)
- Adds randomness and reduces correlation between trees

### Strengths of Random Forest

- ✅ Highly accurate (often best out-of-the-box performance)
- ✅ Resistant to overfitting
- ✅ Handles non-linear relationships
- ✅ Works well with high-dimensional data
- ✅ Provides feature importance rankings
- ✅ Robust to outliers and noise

### Limitations

- ❌ Slower to train and predict than simpler models
- ❌ Less interpretable (black box)
- ❌ Requires more memory
- ❌ Can be overkill for simple problems

### When to Use Random Forest

- You need high accuracy
- Data has complex non-linear patterns
- You have sufficient computational resources
- Feature interactions are important
- You want to know which features matter most

---

## Model 4: Support Vector Machine (SVM) Regression

### The Concept: Finding the Best Fit Tube

Unlike linear regression which tries to minimize distance to a line, SVM regression creates a "tube" around predictions and tries to fit as many data points as possible within this tube.

**Analogy**: Imagine drawing a road with margins. SVM tries to make the road narrow enough to be precise, but wide enough that most houses fit comfortably inside.

### How SVM Regression Works

1. **Define a margin (ε-tube)**: Create a tolerance zone around predictions
2. **Fit data within tube**: Points inside the tube have zero error
3. **Penalize outliers**: Points outside the tube contribute to the error
4. **Find optimal tube**: Balance between tube width and number of violations

**Example**:
- If ε = ₹5 lakhs, and prediction is ₹50 lakhs
- Actual prices between ₹45-55 lakhs are considered "good enough" (zero error)
- Prices outside this range are penalized

### The Kernel Trick

SVM's superpower is handling non-linear relationships through **kernels**:

**Linear Kernel**: Standard straight-line relationships
**RBF (Radial Basis Function) Kernel**: Can model complex curved relationships
- Each data point becomes a "center of influence"
- Nearby points have more impact on predictions
- Creates smooth, flexible boundaries

**Polynomial Kernel**: Models polynomial relationships
- Good for relationships like: Price ∝ (sqft)²

### Key Parameters

**C (Regularization)**: Controls trade-off between simplicity and accuracy
- High C: Fit training data closely (risk overfitting)
- Low C: Prefer simpler model (may underfit)

**epsilon (ε)**: Width of the tube
- Larger ε: More tolerant (simpler model)
- Smaller ε: Less tolerant (tries to fit more precisely)

**kernel**: Type of kernel function
- 'linear': For linear relationships
- 'rbf': Most common, handles non-linearity
- 'poly': For polynomial relationships

### Strengths of SVM

- ✅ Effective in high-dimensional spaces
- ✅ Memory efficient (uses support vectors, not all data)
- ✅ Versatile (different kernels for different data)
- ✅ Works well when features >> samples

### Limitations

- ❌ Slow on large datasets (>10,000 samples)
- ❌ Requires feature scaling
- ❌ Choosing right kernel and parameters is tricky
- ❌ Less interpretable than linear models
- ❌ Sensitive to noisy data

### When to Use SVM

- Medium-sized datasets (hundreds to thousands of samples)
- High-dimensional data
- Clear margin of separation exists
- You need a robust model against outliers

---

## Model 5: K-Nearest Neighbors (KNN) Regression

### The "Ask Your Neighbors" Approach

KNN is beautifully simple: To predict a house price, look at the K most similar houses and average their prices!

**Real-Life Analogy**: You want to price your house. You find the 5 most similar houses in your database (same size, location, bedrooms) and average their prices. That's your estimate!

### How KNN Works

1. **Get new house features**: sqft, bedrooms, bathrooms, location
2. **Calculate distance**: Measure "similarity" to all houses in training data
3. **Find K nearest neighbors**: Select K most similar houses
4. **Average their prices**: That's your prediction!

**Example** (K=5):
- Your house: 1500 sqft, 3 BHK, Koramangala
- 5 most similar houses in data:
  - House A: ₹58 lakhs (1480 sqft, 3 BHK, Koramangala)
  - House B: ₹62 lakhs (1520 sqft, 3 BHK, Koramangala)
  - House C: ₹60 lakhs (1500 sqft, 3 BHK, Koramangala)
  - House D: ₹59 lakhs (1490 sqft, 3 BHK, Koramangala)
  - House E: ₹61 lakhs (1510 sqft, 3 BHK, Koramangala)
- **Prediction**: (58+62+60+59+61)/5 = ₹60 lakhs

### Measuring Distance

**Euclidean Distance** (most common):
- Like measuring straight-line distance on a map
- Distance = √[(sqft₁-sqft₂)² + (bath₁-bath₂)² + ...]

**Manhattan Distance**:
- Like driving through city blocks (only horizontal/vertical)
- Distance = |sqft₁-sqft₂| + |bath₁-bath₂| + ...

### Key Parameters

**n_neighbors (K)**: Number of neighbors to consider
- Small K (e.g., 3): Sensitive to noise, captures local patterns
- Large K (e.g., 20): Smoother predictions, less noise sensitivity
- **Finding right K**: Use cross-validation!

**weights**: How to weight neighbors
- 'uniform': All K neighbors contribute equally
- 'distance': Closer neighbors have more influence

**metric**: Distance calculation method
- 'euclidean': Standard straight-line distance
- 'manhattan': City-block distance

### Strengths of KNN

- ✅ Simple and intuitive
- ✅ No training phase (lazy learner)
- ✅ Naturally handles non-linear relationships
- ✅ Can be very accurate with right K
- ✅ Works for both regression and classification

### Limitations

- ❌ Slow predictions (must calculate all distances)
- ❌ Requires lots of memory (stores all training data)
- ❌ Sensitive to irrelevant features
- ❌ **Curse of dimensionality**: Performance degrades with many features
- ❌ Requires feature scaling

### When to Use KNN

- Small to medium datasets
- Need quick model without training
- Relationships are local (similar inputs → similar outputs)
- Data is low-dimensional (< 20 features)
- Interpretability matters (can explain by showing neighbors)

---

## Model 6: Naive Bayes Regression

### Understanding Naive Bayes

Naive Bayes is typically used for classification, but we can adapt it for regression using **Gaussian Naive Bayes** by discretizing the target variable.

**The "Naive" Assumption**: All features are independent given the target.

**Example**: The model assumes that:
- Number of bedrooms doesn't affect the impact of square footage
- Location is independent of bathrooms
- (This is "naive" because in reality, these ARE related!)

### How Naive Bayes Works

**The Bayesian Approach**:
1. **Prior belief**: What do we know about house prices in general?
2. **New evidence**: What features does this specific house have?
3. **Updated belief**: Combine prior knowledge with new evidence using Bayes' Theorem

**Bayes' Theorem**:
```
P(Price | Features) = P(Features | Price) × P(Price) / P(Features)
```

**In Simple Terms**:
- "Given these features, what's the most likely price range?"
- The model learns probability distributions for each feature
- Combines them (naively assuming independence) for prediction

### For Regression: Gaussian Naive Bayes

To use Naive Bayes for price prediction:
1. **Discretize prices**: Group into ranges (₹20-40L, ₹40-60L, ₹60-80L, etc.)
2. **Calculate probabilities**: For each price range, what's the probability distribution of features?
3. **Predict**: For new house, find most probable price range
4. **Return value**: Use mean of that range

**Alternative**: Use the probabilities as weights to calculate expected value.

### Key Assumptions

1. **Feature Independence**: Features don't influence each other's relationship with price
2. **Gaussian Distribution**: Features follow normal (bell curve) distribution within each class
3. **Sufficient Data**: Need enough examples in each price range

### Strengths of Naive Bayes

- ✅ Very fast training and prediction
- ✅ Works well with limited data
- ✅ Handles high-dimensional data
- ✅ Probabilistic predictions (gives confidence)
- ✅ Simple and interpretable

### Limitations

- ❌ Strong (naive) independence assumption rarely holds
- ❌ Not naturally designed for regression
- ❌ Requires discretization (loses information)
- ❌ Assumes Gaussian distribution
- ❌ Generally less accurate than other regression methods

### When to Use Naive Bayes

- Need very fast predictions
- Limited training data
- High-dimensional data
- Want probabilistic outputs
- Baseline model for comparison
- Real-time applications

**Note**: For house price prediction specifically, Naive Bayes is not the best choice, but understanding it is valuable for your ML toolkit!

---

## Comparing All Six Models

| Model | Accuracy | Speed | Interpretability | Best For |
|-------|----------|-------|-----------------|----------|
| **Linear Regression** | Moderate | ⚡⚡⚡ Fast | 🔍🔍🔍 High | Linear relationships, baseline |
| **Decision Tree** | Good | ⚡⚡ Moderate | 🔍🔍 Medium | Non-linear, interpretable rules |
| **Random Forest** | Excellent | ⚡ Slow | 🔍 Low | High accuracy, robust |
| **SVM** | Excellent | ⚡ Slow | 🔍 Low | High-dimensional, complex patterns |
| **KNN** | Good | ⚡ Slow | 🔍🔍 Medium | Local patterns, small datasets |
| **Naive Bayes** | Moderate | ⚡⚡⚡ Fast | 🔍🔍 Medium | Fast baseline, probabilistic |

### Practical Recommendations

**For House Price Prediction**:
1. **Start with**: Linear Regression (fast baseline)
2. **Try next**: Random Forest (likely best performance)
3. **If high accuracy needed**: SVM with RBF kernel
4. **If interpretability matters**: Decision Tree
5. **If data is limited**: KNN with proper K selection
6. **For quick prototype**: Naive Bayes

**General Strategy**:
- Try multiple models
- Use cross-validation for fair comparison
- Consider trade-offs (accuracy vs. speed vs. interpretability)
- Choose based on your specific requirements

---

## Model Evaluation: How Do We Measure Success?

### The R² Score (R-Squared)

R² tells us: **"What percentage of price variation does my model explain?"**

**Scale**: 0 to 1 (or 0% to 100%)
- **R² = 1.0 (100%)**: Perfect predictions
- **R² = 0.8 (80%)**: Model explains 80% of price variations
- **R² = 0.0 (0%)**: Model no better than just guessing the average

**Example Interpretation**:
- R² = 0.84 means: "Our model accounts for 84% of why prices vary. The remaining 16% is due to factors we haven't captured or random noise."

### What's a Good R² Score?

- **R² > 0.9**: Excellent (but check for overfitting!)
- **R² = 0.7-0.9**: Good for most real-world problems
- **R² = 0.5-0.7**: Moderate (room for improvement)
- **R² < 0.5**: Poor (rethink features or model)

For house prices with many unpredictable factors, **R² ≈ 0.8** is quite good!

---

## K-Fold Cross-Validation: The Ultimate Test

### The Problem with Single Train-Test Split

What if, by chance, your test set contains only easy-to-predict houses? Your model might seem better than it actually is!

**Analogy**: Imagine evaluating a student's knowledge based on just one quiz. One quiz might be unusually easy or hard. Multiple quizzes give a better picture.

### How K-Fold Cross-Validation Works

**K-Fold (K=5 example)**:

1. **Split data into 5 equal parts** (folds)
2. **Iteration 1**: Train on folds 1,2,3,4 → Test on fold 5
3. **Iteration 2**: Train on folds 1,2,3,5 → Test on fold 4
4. **Iteration 3**: Train on folds 1,2,4,5 → Test on fold 3
5. **Iteration 4**: Train on folds 1,3,4,5 → Test on fold 2
6. **Iteration 5**: Train on folds 2,3,4,5 → Test on fold 1
7. **Final Score**: Average of all 5 test scores

### Why This is Better

- **Robustness**: Every data point gets to be in the test set once
- **Reliability**: Multiple evaluations reduce luck factor
- **Confidence**: If all 5 scores are similar (e.g., all around 0.82), you can trust the model
- **Variance Detection**: If scores vary wildly (0.5, 0.9, 0.6, 0.85, 0.7), something's wrong

**In Our Project**: We get 5 different R² scores and can see consistency:
```
[0.81, 0.82, 0.84, 0.83, 0.81]
```
Average ≈ 0.82, with low variance → Trustworthy model!

---

## GridSearchCV: Finding the Best Model Configuration

### The Hyperparameter Problem

Each model has settings (hyperparameters) that affect performance:
- Linear Regression: Should we fit an intercept?
- Lasso: What alpha value?
- Decision Tree: Criterion? Splitter?

**The Question**: Which combination of settings works best?

### What is GridSearchCV?

GridSearchCV systematically tests all combinations of parameters using cross-validation.

**Example**: For Lasso with:
- Alpha options: [1, 2]
- Selection options: ['random', 'cyclic']

**Grid of possibilities**:
1. Alpha=1, Selection='random'
2. Alpha=1, Selection='cyclic'
3. Alpha=2, Selection='random'
4. Alpha=2, Selection='cyclic'

**For each combination**: Run 5-fold cross-validation, get average score.

**Final Result**: "Best combination is Alpha=1, Selection='cyclic' with R²=0.82"

### Why This is Powerful

- ✅ Eliminates guesswork
- ✅ Finds optimal configuration automatically
- ✅ Prevents overfitting to a single train-test split (uses cross-validation)
- ✅ Compares models fairly

---

## Model Performance Comparison

After running GridSearchCV on all six models with optimal parameters, here's how they compare:

| Model | R² Score | Training Time | Prediction Speed | Complexity |
|-------|----------|---------------|------------------|------------|
| Linear Regression | ~0.82 | Fast | Very Fast | Low |
| Decision Tree | ~0.71 | Fast | Very Fast | Medium |
| Random Forest | ~0.85 | Slow | Moderate | High |
| SVM | ~0.80 | Very Slow | Fast | High |
| KNN | ~0.75 | None (Lazy) | Very Slow | Low |
| Naive Bayes | ~0.65 | Very Fast | Very Fast | Low |

**Key Findings**:
- 🏆 **Winner: Random Forest** - Best accuracy (~0.85) but slower
- 🥈 **Runner-up: Linear Regression** - Great balance of simplicity, speed, and accuracy (~0.82)
- 📊 **Most Interpretable**: Decision Tree
- ⚡ **Fastest**: Naive Bayes and Linear Regression
- 🎯 **Best for Production**: Linear Regression or Random Forest (depending on requirements)

**For Our House Price Problem**: We'll use **Linear Regression** for its excellent balance of performance, speed, and interpretability!

---

## Making Predictions: Putting It All Together

### The Prediction Function

Once trained, we can predict prices for new houses. Here's what happens behind the scenes:

**Input**: Location, Square feet, Bathrooms, Bedrooms
**Process**:
1. Find location's column index (remember one-hot encoding?)
2. Create an array with all features = 0
3. Fill in: square feet, bathrooms, bedrooms
4. Set location's column = 1
5. Pass to model: model.predict([feature_array])

**Output**: Predicted price in lakhs

### Example Prediction

**Input**: "1st Phase JP Nagar", 1000 sq ft, 2 bathrooms, 2 bedrooms

```python
predict_price('1st Phase JP Nagar', 1000, 2, 2)
# Output: 83.5 (₹83.5 lakhs)
```

**What just happened?**
The model used the patterns it learned from 7,000+ houses to estimate that a 2 BHK, 1000 sq ft house in JP Nagar should cost around ₹83.5 lakhs.

---

## Model Persistence: Saving Your Work

### Why Save Models?

Training takes time and computational resources. Once you have a good model:
- **Save it** so you don't have to retrain every time
- **Deploy it** in applications (websites, mobile apps)
- **Share it** with others

### Pickle: Python's Serialization Tool

**Pickle** converts Python objects into a byte stream that can be saved to disk.

```python
import pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(trained_model, f)
```

**Later, load it back**:
```python
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
    # Now you can use model.predict() without retraining!
```

### Saving Column Information

The model also needs to know which columns exist and their order. Save this metadata:

```python
import json
columns = {'data_columns': ['total_sqft', 'bath', 'bhk', 'location_1', ...]}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
```

**Why?** When someone uses your model later, they'll know exactly which features to provide and in what order.

---

## Hands-On: Code Implementation

Now let's see how we implement these concepts in code!

### Step 1: One-Hot Encoding Locations

```python
dummies = pd.get_dummies(df10.location) #One-hot encoding
dummies.head(3)
```

Combine with original data:

```python
df11 = pd.concat([df10, dummies.drop('other',axis="columns")],axis='columns')

df11.head()
```

Remove original location column:

```python
df12 = df11.drop('location', axis="columns")

df12.head()
```

### Step 2: Separate Features and Target

```python
X = df12.drop(['price'],axis='columns')
y = df12.price
```

### Step 3: Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
```

**What this does**:
- `test_size=0.2`: 20% for testing, 80% for training
- `random_state=10`: Ensures reproducible splits (same random split each time)

### Step 4: Train Linear Regression

```python
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

lr_clf.score(X_test, y_test)
```

The `score()` method returns the R² value.

### Step 5: Cross-Validation for Linear Regression

```python
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)
```

**Output**: Array of 5 R² scores, one for each fold.

### Step 6: GridSearchCV to Compare All Models

```python
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 10, 20]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'svm': {
            'model': SVR(),
            'params': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1, 1]
            }
        },
        'knn': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }
    
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    
    for algo_name, config in algos.items():
        print(f"Training {algo_name}...")
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False, n_jobs=-1)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
        print(f"{algo_name}: {gs.best_score_:.4f}")

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params']).sort_values(by='best_score', ascending=False)

# Run the comparison
results = find_best_model_using_gridsearchcv(X, y)
print("\n" + "="*50)
print("FINAL RESULTS (sorted by score):")
print("="*50)
print(results)
```

**What this does**:
1. Defines six models with different parameter options
2. For each model, tries all parameter combinations
3. Uses 5-fold cross-validation for each combination
4. Returns results sorted by best score
5. Uses parallel processing (n_jobs=-1) for speed

**Note**: Random Forest and SVM may take several minutes to run due to parameter combinations!

### Step 7: Create Prediction Function

```python
#Now for testing, how can we handle the location column?

X.columns
```

Find location column index:

```python
np.where(X.columns=='2nd Phase Judicial Layout')[0][0] #This will return the index number
```

Complete prediction function:

```python
def predict_price(location, sqft, bath, bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
```

### Step 8: Test the Model

```python
predict_price('1st Phase JP Nagar', 1000, 2, 2)
```

### Step 9: Export the Model

```python
import pickle
with open('banglore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)
```

### Step 10: Export Column Information

```python
import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
```

---

### 📥 Download Material

- 📓 Download Notebook:  
  [banglore_home_prices_final.ipynb](../files/day1/banglore_home_prices_final.ipynb)

- 📊 Download Dataset (CSV):  
  [bengaluru_house_prices.csv](../files/day1/bengaluru_house_prices.csv)

Run the cells, experiment with different parameters, and see how model performance changes!

---

## Key Takeaways

Today you learned about **six powerful machine learning models**:

1. **Train-Test Split**: Always evaluate models on unseen data to avoid overfitting
2. **Linear Regression**: Simple, interpretable baseline - great for linear relationships
3. **Decision Trees**: Interpretable rules, handles non-linearity, prone to overfitting
4. **Random Forest**: Ensemble of trees, typically highest accuracy, "wisdom of crowds"
5. **SVM (Support Vector Machine)**: Effective in high dimensions, kernel trick for non-linearity
6. **KNN (K-Nearest Neighbors)**: Instance-based learning, simple but memory-intensive
7. **Naive Bayes**: Fast probabilistic model, works with limited data
8. **R² Score**: Measures how much price variation your model explains (aim for > 0.7)
9. **K-Fold Cross-Validation**: Robust evaluation using multiple train-test splits
10. **GridSearchCV**: Automated hyperparameter tuning with cross-validation
11. **Model Persistence**: Save trained models using pickle for reuse

**Most Important Lessons**:
- 🎯 **No single "best" model** - it depends on your specific problem and constraints
- ⚖️ **Trade-offs matter**: Accuracy vs. Speed vs. Interpretability vs. Memory
- 🔬 **Always experiment**: Try multiple models, use cross-validation for fair comparison
- 📊 **Context is key**: Choose based on data size, feature count, and business requirements
- 🚀 **Start simple**: Begin with Linear Regression, then try more complex models if needed

---

## What's Next?

You now have a complete end-to-end machine learning pipeline:
1. ✅ Data Cleaning
2. ✅ Feature Engineering
3. ✅ Model Training
4. ✅ Model Evaluation
5. ✅ Making Predictions

**Next Steps**:
- Deploy this model as a web application
- Try more advanced models (Random Forests, Gradient Boosting)
- Collect more features (proximity to schools, crime rates, etc.)
- Apply these techniques to different problems

Congratulations on completing this comprehensive machine learning project! 🎉
