import pandas as pd
import numpy as np

url = 'https://github.com/YBI-Foundation/Dataset/raw/main/Insurance%20Premium.csv' #Loading external dataset
df = pd.read_csv(url)

df.head()

# Checks data types and non-null values
df.info()

# Summary statistics for numeric features
df.describe()

# Inspect categories in text columns
print("Gender:\n", df['Gender'].value_counts())
print("\nSmoker:\n", df['Smoker'].value_counts())
print("\nRegion:\n", df['Region'].value_counts())
# Mapping categories to integers
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
df['Smoker'] = df['Smoker'].map({'no': 0, 'yes': 1})
df['Region'] = df['Region'].map({'north': 0, 'east': 1, 'south': 2, 'west': 3})

df.head()

from sklearn.preprocessing import StandardScaler

# Define target variable (y) and features (X)
y = df['Premium']
X = df[['Age', 'Gender', 'BMI', 'Children', 'Smoker', 'Region']].copy()

# Standardize continuous variables (Age, BMI)
sc = StandardScaler()
X[['Age', 'BMI']] = sc.fit_transform(X[['Age', 'BMI']])

X.head()

from sklearn.model_selection import train_test_split

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2529
)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

from sklearn.ensemble import RandomForestRegressor

# Initialize and train the regressor
rfr = RandomForestRegressor(random_state=2529)
rfr.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Generate predictions on the test set
y_pred = rfr.predict(X_test)

# Print performance metrics
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='teal', alpha=0.6)
plt.xlabel("Actual Premium")
plt.ylabel("Predicted Premium")
plt.title("Actual vs. Predicted Insurance Premium")
plt.grid(True)
plt.show()

# Sample input vector: [Age (scaled), Gender, BMI (scaled), Children, Smoker, Region]
X_new = np.array([[-1.22516069, 0, 1.44959597, 0, 0, 2]])

# Make prediction
y_pred_new = rfr.predict(X_new)
print(f"Predicted Premium: ${y_pred_new[0]:,.2f}")