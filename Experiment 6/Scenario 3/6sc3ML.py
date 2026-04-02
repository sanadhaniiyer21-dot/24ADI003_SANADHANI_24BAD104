
# ─────────────────────────────────────────
# STEP 1 – Import Libraries
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# STEP 2 – Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv("C:\\Users\\sanad\\Downloads\\income_random_forest.csv")

# ─────────────────────────────────────────
# STEP 3 – Features & Target
# ─────────────────────────────────────────
X = df[['Age', 'EducationYears', 'HoursPerWeek', 'Experience']]
y = df['Income']

# ─────────────────────────────────────────
# STEP 4 – Train-Test Split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ─────────────────────────────────────────
# STEP 5 – Scaling
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# STEP 6 – Accuracy vs Number of Trees (FIXED)
# ─────────────────────────────────────────
tree_range = [10, 50, 100, 150, 200]

train_acc = []
test_acc  = []

for n in tree_range:
    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    train_acc.append(accuracy_score(y_train, model.predict(X_train_scaled)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test_scaled)))

# ─────────────────────────────────────────
# STEP 6.5 – Plot BOTH Train & Test Accuracy
# ─────────────────────────────────────────
plt.clf()
plt.close('all')

plt.figure()

plt.plot(tree_range, train_acc, marker='o', label='Train Accuracy')
plt.plot(tree_range, test_acc,  marker='o', label='Test Accuracy')

plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees')

plt.legend()
plt.grid(True)

plt.show()

# ─────────────────────────────────────────
# STEP 7 – Train Final Model
# ─────────────────────────────────────────
best_n = tree_range[np.argmax(test_acc)]

rf_model = RandomForestClassifier(
    n_estimators=best_n,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train_scaled))
rf_test_acc  = accuracy_score(y_test,  rf_model.predict(X_test_scaled))

print("\nBest number of trees:", best_n)
print("Random Forest - Train Accuracy:", rf_train_acc)
print("Random Forest - Test Accuracy :", rf_test_acc)

# ─────────────────────────────────────────
# STEP 8 – Feature Importance
# ─────────────────────────────────────────
features = list(X.columns)
importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# Plot
x = np.arange(len(importance_df))

plt.figure(figsize=(8, 5))
plt.bar(x, importance_df['Importance'])

plt.xticks(x, importance_df['Feature'], rotation=20)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance (Random Forest)')

plt.tight_layout()
plt.show()
