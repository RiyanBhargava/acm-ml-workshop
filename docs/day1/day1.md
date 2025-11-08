# Day 1: Data Cleaning and Feature Engineering

## Introduction to Real Estate Price Prediction

Welcome to Day 1 of our Machine Learning workshop! Today, we'll embark on an exciting journey to build a real estate price prediction model using data from Bangalore, India. Before we dive into coding, let's understand the fundamental concepts that make machine learning projects successful.

## Understanding the Problem

Imagine you're a real estate agent or a home buyer trying to determine the fair price of a property. What factors would you consider? The location, size of the house, number of bedrooms, bathrooms, and many other features play crucial roles. Our goal is to teach a computer to understand these patterns and predict prices automatically.

## The Machine Learning Pipeline

Every successful machine learning project follows a structured approach:

1. **Data Collection**: Gathering relevant information
2. **Data Cleaning**: Removing errors and inconsistencies
3. **Feature Engineering**: Creating meaningful variables from raw data
4. **Exploratory Data Analysis**: Understanding patterns in data
5. **Model Building**: Training algorithms (we'll cover this in Day 2)
6. **Model Evaluation**: Testing how well our model performs

Today, we'll focus on the first four crucial steps - the foundation of any ML project.

---

## 1. Data Cleaning: The Foundation of Quality Models

### Why is Data Cleaning Important?

Think of data cleaning like preparing ingredients before cooking. You wouldn't use rotten vegetables or unwashed produce in a meal, right? Similarly, dirty data leads to poor predictions. Real-world data is messy - it has missing values, inconsistencies, duplicates, and errors that can mislead our model.

### Understanding Missing Values

**Example Scenario**: Imagine a dataset of house listings where some entries don't have information about the number of bathrooms or the location. What should we do?

**Two Common Approaches**:
1. **Deletion**: Remove rows with missing data (when dataset is large)
2. **Imputation**: Fill missing values with mean, median, or mode (when data is scarce)

**When to delete vs. impute?** If you have 13,000 rows and only 1,000 have missing values, deletion is safe. But if 8,000 rows have missing values, you might want to impute to preserve information.

### Dealing with Irrelevant Features

Not every piece of information is useful. Consider these columns in a house price dataset:
- **Area Type**: The type of area measurement (built-up, plot area, etc.)
- **Society Name**: The specific housing society
- **Balcony Count**: Number of balconies
- **Availability**: When the house is available

**Question**: Do these strongly influence price predictions? Often, the answer is no. Removing irrelevant features:
- Simplifies the model
- Reduces computational cost
- Prevents overfitting
- Improves model performance

### Standardizing Data Formats

**The Problem**: Your dataset has a "size" column with values like:
- "2 BHK"
- "3 Bedroom"
- "4 BHK"

**The Solution**: Extract just the numeric part (2, 3, 4) to create a consistent "bhk" (Bedroom, Hall, Kitchen) column that machines can understand.

### Handling Range Values

**Example**: A property's size is listed as "1133 - 1384 sq ft" instead of a single number.

**Solution**: Convert ranges to their average. For "1133 - 1384", we'd use (1133 + 1384) / 2 = 1258.5 sq ft.

### Cleaning Inconsistent Units

Sometimes you'll find values like:
- "2500 sq ft"
- "34.46 Sq. Meter"
- "4125 Perch"

These mixed units make comparison impossible. The best approach is to convert everything to a standard unit or exclude entries that can't be converted reliably.

---

## 2. Feature Engineering: Creating Meaningful Variables

Feature engineering is the art of creating new, more informative variables from existing data. It's often the difference between a mediocre and an excellent model.

### Creating Price Per Square Foot

**Why?** Absolute price doesn't tell the whole story. A 3000 sq ft house costing ₹60 lakhs might be a better deal than a 1000 sq ft house at ₹30 lakhs.

**Calculation**: Price per sq ft = (Price × 100,000) / Total Square Feet

This normalized metric helps us compare properties of different sizes on equal footing.

### Grouping Rare Categories

**The Problem**: Your dataset has 1,293 unique locations, but 1,052 of them appear fewer than 10 times.

**The Solution**: Group infrequent categories into an "other" category. Why?

1. **Statistical Significance**: Locations with only 1-2 properties don't provide enough data for reliable patterns
2. **Model Simplicity**: Fewer categories mean fewer variables to process
3. **Generalization**: Helps the model focus on common patterns rather than rare exceptions

**Real-world analogy**: If you're learning to recognize cars, you'd focus on common brands like Toyota, Honda, and Ford before worrying about rare vintage models.

---

## 3. Outlier Detection and Removal

Outliers are extreme values that don't fit the general pattern. They can severely distort your model's understanding of the data.

### What are Outliers?

**Example 1**: A 6-bedroom house with only 1,020 square feet total. That's roughly 170 sq ft per room - smaller than most bathrooms! This is clearly an error or exceptional case.

**Example 2**: A property listed at ₹12,000,000 per square foot when most properties in that area are ₹5,000-10,000 per sq ft.

### Why Remove Outliers?

Imagine teaching someone about typical house prices by showing them:
- 99 normal houses (₹30-80 lakhs)
- 1 ultra-luxury mansion (₹500 lakhs)

They might develop a skewed understanding. Similarly, outliers can mislead machine learning models.

### Domain-Based Outlier Removal

**Rule of Thumb**: In urban Indian housing, a reasonable minimum is about 300 square feet per bedroom.

**Logic**: 
- 1 BHK should have at least 300 sq ft
- 2 BHK should have at least 600 sq ft
- 3 BHK should have at least 900 sq ft

Properties below these thresholds are likely data entry errors or exceptional cases we should exclude.

### Outlier Removal using Box Plots and IQR

#### Box Plot Visualization

A box plot (or whisker plot) is a graphical representation that helps visualize the spread and skewness of numerical data.
It displays:

- **Median (Q2)** — The middle value of the dataset.
- **First Quartile (Q1)** — The 25th percentile (lower quartile).
- **Third Quartile (Q3)** — The 75th percentile (upper quartile).
- **Interquartile Range (IQR)** — The difference between Q3 and Q1 (IQR = Q3 - Q1).
- **Whiskers and Points** — Points outside the whiskers represent potential outliers.

A box plot makes it easy to visually identify outliers — these are the points that appear outside the whiskers (i.e., far from the main cluster of data).

#### Interquartile Range (IQR) Method
The IQR method is a statistical technique used to detect and remove outliers.

**Steps:**

1. Compute Q1 and Q3 — Find the 25th and 75th percentiles of the data.

2. Calculate IQR

    ```𝐼𝑄𝑅=𝑄3−𝑄1```

3. Determine cutoff limits

    ```Lower bound = Q1 - 1.5 × IQR```

    ```Upper bound = Q3 + 1.5 × IQR```

4. Identify and remove outliers — Any value less than the lower bound or greater than the upper bound is considered an outlier.

## 4. Preparing Data for Machine Learning

### One-Hot Encoding: Converting Categories to Numbers

**The Challenge**: Machine learning algorithms work with numbers, not text. How do we handle the "location" column?

**Example**: You have three locations:
- Rajaji Nagar
- Hebbal
- Koramangala

**One-Hot Encoding Solution**: Create separate binary (0 or 1) columns for each location:

| Price | BHK | Rajaji_Nagar | Hebbal | Koramangala |
|-------|-----|--------------|--------|-------------|
| 50    | 2   | 1            | 0      | 0           |
| 75    | 3   | 0            | 1      | 0           |
| 60    | 2   | 0            | 0      | 1           |

**The "Other" Category**: We don't create a column for "other" because if all location columns are 0, the model knows it's "other."

### Separating Features and Target

**Features (X)**: The input variables we use to make predictions
- Total square feet
- Number of bathrooms
- Number of bedrooms (BHK)
- Location (one-hot encoded)

**Target (y)**: What we're trying to predict
- Price

This separation is crucial because we train the model to find patterns between X and y.

---

## Data Visualization: Seeing Patterns

Visualization helps us understand our data intuitively.

### Histogram of Price Per Square Foot

A histogram shows the distribution of values:
- **X-axis**: Price ranges (e.g., ₹3,000-4,000, ₹4,000-5,000)
- **Y-axis**: How many properties fall in each range

**What to look for**:
- Where most properties are concentrated
- Whether the distribution is normal (bell-shaped)
- Presence of extreme values

### Scatter Plots for Outlier Detection

**Purpose**: Compare 2 BHK vs. 3 BHK properties in the same location.

**Axes**:
- X-axis: Total square feet
- Y-axis: Price

**What we expect**: 
- 3 BHK properties (green) should generally be above 2 BHK properties (blue) for the same square footage
- Both should show an upward trend (more sq ft = higher price)

**Red Flags**:
- Blue dots (2 BHK) above green crosses (3 BHK) at the same square footage
- Properties that don't follow the general upward trend

---

## Summary of Day 1 Concepts

Today we've learned that successful machine learning requires careful preparation:

1. **Clean your data**: Remove inconsistencies, handle missing values, standardize formats
2. **Engineer features**: Create meaningful variables like price per sq ft
3. **Remove outliers**: Eliminate extreme values using domain knowledge and statistical methods
4. **Prepare for algorithms**: Convert categories to numbers, separate features from targets

**Key Takeaway**: "Garbage in, garbage out." The quality of your data directly determines the quality of your predictions. Spending time on data cleaning and feature engineering is not optional - it's essential.

Tomorrow, we'll take this cleaned dataset and build machine learning models to predict house prices!

---

## Hands-On: Code Implementation

Now that you understand the theory, let's see how we implement these concepts in Python!

### Step 1: Import Libraries

```python
#Importing the necessary libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
```

**What we're importing**:
- `pandas`: For data manipulation
- `numpy`: For numerical operations
- `matplotlib`: For creating visualizations

### Step 2: Load the Dataset

```python
df1 = pd.read_csv("bengaluru_house_prices.csv")
df1.head()
```

This loads our CSV file and displays the first few rows.

### Step 3: Understand Dataset Dimensions

```python
df1.shape #For knowing number of rows and columns
```

### Step 4: Explore Area Types

```python
#To know the different types of elements in the column 'area_types' we use the below code:-
df1.groupby('area_type')['area_type'].agg('count')
```

### Step 5: Drop Irrelevant Columns

```python
#Assuming "area_type", "society", "balcony", "availability" to be not very important, so we can drop those columns

df2 = df1.drop(["area_type", "society", "balcony", "availability"],axis='columns')

df2.head()
```

### Step 6: Check for Missing Values

```python
#Now to know at which places in our dataset, there are empty/NA values,
df2.isnull().sum()
```

### Step 7: Remove Missing Values

```python
df3 = df2.dropna()

df3.isnull().sum() #This shows that now there are no empty elements in our dataset
```

### Step 8: Clean the 'Size' Column

```python
df3['size'].unique() #This code will return all the unique values in the column 'size'
```

Extract numeric bedroom count:

```python
df3['bhk'] = df3['size'].str.split(' ').str[0].astype(int)
```

### Step 9: Handle Range Values in 'total_sqft'

```python
df3.total_sqft.unique()
```

Check for non-float values:

```python
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df3[~df3['total_sqft'].apply(is_float)]
```

Convert range values to averages:

```python
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
```

### Step 10: Feature Engineering - Price Per Square Foot

```python
df5 = df4.copy()

#Now we will be adding a new column which will be price per sq ft, for later purposes of outlier detection

df5['price_per_sqft'] = df5['price']*100000 / df5['total_sqft']

df5.head()
```

### Step 11: Clean Location Data

```python
len(df5.location.unique())
```

Remove whitespace and group rare locations:

```python
df5.location = df5.location.apply(lambda x: x.strip())
#This code removes leading and trailing whitespace from each value in the 'location' column

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)

location_stats
```

Group locations with fewer than 10 occurrences:

```python
len(location_stats[location_stats <= 10]) #We can observe that there are 1052 locations whose repetition is less than 10 out of 1293 locations...So we can classify them as 'Others'

# Assuming location_stats is a Series like: location -> count
location_stats_less_than_10 = location_stats[location_stats <= 10].index  # use .index

# Now replace rare locations
df5['location'] = df5['location'].where(~df5['location'].isin(location_stats_less_than_10), 'other')

len(df5.location.unique())
```

### Step 12: Remove Outliers Based on Square Feet per Bedroom

```python
df6 = df5[~(df5.total_sqft/df5.bhk<300)]

df6.shape
```

### Step 13: Remove Outliers Using Box plots and IQR Method
- Box plots to check outliers

```
plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
sns.boxplot(df6['bath'])
plt.title('Bathrooms')

plt.subplot(1,3,2)
sns.boxplot(df6['bhk'])
plt.title('BHK')

plt.subplot(1,3,3)
sns.boxplot(df6['price_per_sqft'])
plt.title('price_per_sqft')

plt.tight_layout()
plt.show()
```
- Removing outliers
```python
# Remove outliers using IQR for bath, bhk, price_per_sqft
def remove_iqr_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

df6_no_bath_outliers = remove_iqr_outliers(df6, 'bath')
df6_no_bhk_outliers = remove_iqr_outliers(df6_no_bath_outliers, 'bhk')
df7 = remove_iqr_outliers(df6_no_bhk_outliers, 'price_per_sqft')

print("Shape after outlier removal:", df7.shape)

```
- This ensures that the dataset is cleaner and more consistent, improving the accuracy and robustness of further data analysis or model training.

### Step 14 : Performing one hot encoding
```python
dummies = pd.get_dummies(df7.location) #One-hot encoding
dummies.head(3)

df8 = pd.concat([df7, dummies.drop('other',axis="columns")],axis='columns')
df8.head()

df9 =df8.drop('location', axis="columns")
df9.head()
```

### Step 15 : Download the preprocessed dataset and be ready for the next stage of model training
```python
df9.to_csv('preprocessed_data.csv', index=False)
from google.colab import files
files.download('preprocessed_data.csv')
```

---

### 📥 Download Material

- 📓 Download Notebook:  
  [banglore_home_prices_final.ipynb](../files/day1/banglore_home_prices_final.ipynb)

- 📊 Download Dataset (CSV):  
  [bengaluru_house_prices.csv](../files/day1/bengaluru_house_prices.csv)


Run each cell step by step to see data cleaning and feature engineering in action!

---

## What's Next?

In **Day 2**, we'll take this beautifully cleaned dataset and:
- Build machine learning models
- Compare different algorithms
- Evaluate model performance
- Make actual price predictions

See you nect week! 🚀
