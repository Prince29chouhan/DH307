import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

# Load the Excel file
excel_file = 'NFHS_5_India_Districts_Factsheet_Data-1.xlsx'
df = pd.read_excel(excel_file)

#Replace *  values and other non-numerical values with NaN
df.replace(['*', '(', ')', '-'], '', regex=True, inplace=True)

# List of columns to convert to numeric, excluding non-numeric columns like District Names, State/UT
cols_to_numeric = df.columns.drop(['District Names', 'State/UT'])

# Convert columns to numeric, coercing errors to NaN
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Impute missing values using the mean of each column
df.fillna(df.mean(), inplace=True)

# Define the target variable name based on the actual column name in your DataFrame
cervical_cancer_screening_rate_column_name = 'Women (age 30-49 years) Ever undergone a screening test for cervical cancer (%)'

# Rename the column to 'cervical_cancer_screening_rate'
df.rename(columns={cervical_cancer_screening_rate_column_name: 'cervical_cancer_screening_rate'}, inplace=True)

#Prepare the data
X = df.drop(['District Names', 'State/UT', 'cervical_cancer_screening_rate'], axis=1) # Features
y_regression = df['cervical_cancer_screening_rate']

#Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)

#Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#1. Regression (Predicting a Number)
#Models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

#Training
linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

#Predictions
linear_pred = linear_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

#Evaluation
print("Regression Models:")
print("Linear Regression - MSE:", mean_squared_error(y_test, linear_pred), "R2:", r2_score(y_test, linear_pred))
print("Random Forest - MSE:", mean_squared_error(y_test, rf_pred), "R2:", r2_score(y_test, rf_pred))
print("Gradient Boosting - MSE:", mean_squared_error(y_test, gb_pred), "R2:", r2_score(y_test, gb_pred))

#2. Classification (Predicting a Category/Bin)

#Binning
bins = [0, 10, 30, 50, 70, 100] # Example bins - adjust as needed based on the distribution of your target variable
labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
df['screening_category'] = pd.cut(df['cervical_cancer_screening_rate'], bins=bins, labels=labels, right=False) # right=False means the left edge is inclusive

#Data Preparation
X_classification = df.drop(['District Names', 'State/UT', 'cervical_cancer_screening_rate', 'screening_category'], axis=1)
y_classification = df['screening_category']

#Splitting
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

#Scaling
X_train_class = scaler.fit_transform(X_train_class)
X_test_class = scaler.transform(X_test_class)

#Models
logistic_model = LogisticRegression(random_state=42, solver='liblinear', multi_class='ovr') #Good starting point
rf_model_class = RandomForestClassifier(random_state=42)
gb_model_class = GradientBoostingClassifier(random_state=42)
nn_model_class = MLPClassifier(random_state=42, max_iter=300)

#Training
logistic_model.fit(X_train_class, y_train_class)
rf_model_class.fit(X_train_class, y_train_class)
gb_model_class.fit(X_train_class, y_train_class)
nn_model_class.fit(X_train_class, y_train_class)

#Predictions
logistic_pred_class = logistic_model.predict(X_test_class)
rf_pred_class = rf_model_class.predict(X_test_class)
gb_pred_class = gb_model_class.predict(X_test_class)
nn_pred_class = nn_model_class.predict(X_test_class)

#Evaluation
print("\nClassification Models:")
print("Logistic Regression - Accuracy:", accuracy_score(y_test_class, logistic_pred_class))
print(classification_report(y_test_class, logistic_pred_class))
print("Random Forest - Accuracy:", accuracy_score(y_test_class, rf_pred_class))
print(classification_report(y_test_class, rf_pred_class))
print("Gradient Boosting - Accuracy:", accuracy_score(y_test_class, gb_pred_class))
print(classification_report(y_test_class, gb_pred_class))
print("Neural Network - Accuracy:", accuracy_score(y_test_class, nn_pred_class))
print(classification_report(y_test_class, nn_pred_class))

# 3. Risk Scoring (Predicting a Risk Score):  Probability of being in the "Very Low" category.
# Data Preparation: Use the same X_classification as above.  y is now a binary variable: 1 if "Very Low", 0 otherwise
df['high_risk'] = (df['screening_category'] == 'Very Low').astype(int) #1 if very low, 0 otherwise.  Adjust as needed

X_risk = df.drop(['District Names', 'State/UT', 'cervical_cancer_screening_rate', 'screening_category', 'high_risk'], axis=1)
y_risk = df['high_risk']

# Split data
X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)

# Scale data
X_train_risk = scaler.fit_transform(X_train_risk)
X_test_risk = scaler.transform(X_test_risk)

# Models: Logistic Regression and Random Forest are well-suited for probability estimation
logistic_model_risk = LogisticRegression(random_state=42, solver='liblinear')
rf_model_risk = RandomForestClassifier(random_state=42)
gb_model_risk = GradientBoostingClassifier(random_state=42)

# Training
logistic_model_risk.fit(X_train_risk, y_train_risk)
rf_model_risk.fit(X_train_risk, y_train_risk)
gb_model_risk.fit(X_train_risk, y_train_risk)

# Predict probabilities
logistic_proba_risk = logistic_model_risk.predict_proba(X_test_risk)[:, 1]  # Probability of being in the 'high_risk' (Very Low) category
rf_proba_risk = rf_model_risk.predict_proba(X_test_risk)[:, 1]
gb_proba_risk = gb_model_risk.predict_proba(X_test_risk)[:, 1]

# Evaluation (using AUC-ROC as a good metric for probability estimates)
from sklearn.metrics import roc_auc_score

print("\nRisk Scoring Models:")
print("Logistic Regression - AUC:", roc_auc_score(y_test_risk, logistic_proba_risk))
print("Random Forest - AUC:", roc_auc_score(y_test_risk, rf_proba_risk))
print("Gradient Boosting - AUC:", roc_auc_score(y_test_risk, gb_proba_risk))
