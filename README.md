# ybi-internship
repo for project submission of internship done
# Insurance Premium Prediction

A machine learning project to predict insurance premium amounts using demographic and health feature data. 

> **Note:** This project is created and submitted as part of a 15-Day Machine Learning Internship.

---

## 📌 Overview
The objective of this project is to build a regression model that accurately predicts medical insurance premium costs based on individual customer attributes.

## 🛠️ Tech Stack & Dependencies
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`

---

## 📊 Dataset & Features
The dataset is loaded directly from the [YBI Foundation Repository](https://github.com/YBI-Foundation/Dataset/raw/main/Insurance%20Premium.csv).

* **Features:**
  * `Age` (Continuous)
  * `Gender` (Categorical: Male, Female)
  * `BMI` (Continuous)
  * `Children` (Discrete)
  * `Smoker` (Categorical: Yes, No)
  * `Region` (Categorical: North, East, South, West)
* **Target Variable:** `Premium` (Continuous)

---

## ⚙️️ Project Workflow

1. **Exploratory Data Analysis (EDA):** Inspected data types, non-null counts, and category value distribution.
2. **Data Encoding:** Categorical values mapped to numeric integers:
   * `Gender`: Male (0), Female (1)
   * `Smoker`: No (0), Yes (1)
   * `Region`: North (0), East (1), South (2), West (3)
3. **Feature Scaling:** Applied `StandardScaler` on continuous numerical features (`Age`, `BMI`).
4. **Train-Test Split:** Split the dataset into 80% training and 20% testing sets (`random_state=2529`).
5. **Model Training:** Trained a `RandomForestRegressor` model.
6. **Evaluation:** Evaluated performance using MSE, MAE, R² Score, and visual scatter plots.

---

## 📈 Model Performance
Model predictions were evaluated against actual test outcomes:
* **Metrics Used:** Mean Squared Error (MSE), Mean Absolute Error (MAE), and $R^2$ Score.
* **Visualization:** Actual vs. Predicted scatter plot using `matplotlib`.

---

## 🚀 How to Run

1. Clone the repository:
   bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
2. Install required libraries
   bash
   pip install pandas numpy scikit-learn matplotlib
3. Run the script or Jupyter notebook to reproduce the results.