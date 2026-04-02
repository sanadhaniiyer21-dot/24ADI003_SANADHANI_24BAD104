print("Sanadhani - 24BAD104")
# ─────────────────────────────────────────
# STEP 1 – Import Libraries
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# STEP 2 – Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv('heart_stacking.csv')   # <-- change filename

print(df.info())

# ─────────────────────────────────────────
# STEP 3 – Select Features & Target
# ─────────────────────────────────────────
X = df[['Age', 'Cholesterol', 'MaxHeartRate']]
y = df['HeartDisease']

# ─────────────────────────────────────────
# STEP 4 – Train-Test Split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ─────────────────────────────────────────
# STEP 5 – Scaling (IMPORTANT for SVM & LR)
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# STEP 6 – Base Models (Regularized)
# ─────────────────────────────────────────
lr = LogisticRegression(C=0.5, max_iter=1000)   # regularization
svm = SVC(kernel='rbf', C=1, probability=True)  # needed for stacking
dt = DecisionTreeClassifier(max_depth=4, min_samples_split=10)

# Train individually
lr.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)

# Predictions
lr_pred = lr.predict(X_test_scaled)
svm_pred = svm.predict(X_test_scaled)
dt_pred = dt.predict(X_test_scaled)

# Accuracy
lr_acc = accuracy_score(y_test, lr_pred)
svm_acc = accuracy_score(y_test, svm_pred)
dt_acc = accuracy_score(y_test, dt_pred)

print("\nIndividual Model Accuracies:")
print("Logistic Regression:", lr_acc)
print("SVM:", svm_acc)
print("Decision Tree:", dt_acc)

# ─────────────────────────────────────────
# STEP 7 – Stacking Classifier
# ─────────────────────────────────────────
estimators = [
    ('lr', lr),
    ('svm', svm),
    ('dt', dt)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    passthrough=False
)

stack_model.fit(X_train_scaled, y_train)

stack_pred = stack_model.predict(X_test_scaled)
stack_acc = accuracy_score(y_test, stack_pred)

print("Stacking Classifier:", stack_acc)

# ─────────────────────────────────────────
# STEP 8 – Model Comparison Bar Chart
# ─────────────────────────────────────────
models = ['Logistic Regression', 'SVM', 'Decision Tree', 'Stacking']
accuracies = [lr_acc, svm_acc, dt_acc, stack_acc]

plt.clf()
plt.close('all')

plt.figure()

plt.bar(models, accuracies)

plt.ylabel('Accuracy')
plt.title('Model Comparison')

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()
