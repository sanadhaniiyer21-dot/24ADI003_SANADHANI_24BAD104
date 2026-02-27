print("EXP - 5")
print("Name: Sanadhani")
print("Roll No: 24BAD104")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree



# Loan Prediction (Decision Tree)


# Update the path to where loan dataset is located on your system
df = pd.read_csv("loan.csv.csv")

# Fill missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Feature selection
X = df[["ApplicantIncome", "LoanAmount", "Credit_History", "Education", "Property_Area"]]
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Metrics
acc_tree = accuracy_score(y_test, pred)
report_tree = classification_report(y_test, pred)

# Confusion Matrix
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True)
plt.title("Loan Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Tree visualization
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()

# Feature importance plot
feat = pd.Series(model.feature_importances_, index=X.columns)
feat.plot(kind="bar")
plt.title("Feature Importance")
plt.show()
