
# ─────────────────────────────────────────
# STEP 1 – Import Libraries
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ─────────────────────────────────────────
# STEP 2 – Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv('diabetes_bagging.csv')

# ─────────────────────────────────────────
# STEP 3 – Split Dataset
# ─────────────────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ─────────────────────────────────────────
# STEP 4 – Scale the Dataset
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# STEP 5 – Regularized Decision Tree
# ─────────────────────────────────────────
dt_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

# ─────────────────────────────────────────
# STEP 6 – Bagging with Regularized Trees
# ─────────────────────────────────────────
base_tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt'
)

bag_model = BaggingClassifier(
    estimator=base_tree,
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    random_state=42
)

bag_model.fit(X_train_scaled, y_train)
bag_pred = bag_model.predict(X_test_scaled)

# ─────────────────────────────────────────
# STEP 7 – Train & Test Accuracy
# ─────────────────────────────────────────
dt_train_acc  = accuracy_score(y_train, dt_model.predict(X_train_scaled))
dt_test_acc   = accuracy_score(y_test, dt_pred)

bag_train_acc = accuracy_score(y_train, bag_model.predict(X_train_scaled))
bag_test_acc  = accuracy_score(y_test, bag_pred)

print("\nDecision Tree  - Train Accuracy:", dt_train_acc)
print("Decision Tree  - Test Accuracy :", dt_test_acc)
print("Bagging        - Train Accuracy:", bag_train_acc)
print("Bagging        - Test Accuracy :", bag_test_acc)

# ─────────────────────────────────────────
# STEP 8 – Accuracy Comparison Bar Graph
# ─────────────────────────────────────────
models    = ['Decision Tree', 'Bagging']
train_acc = [dt_train_acc, bag_train_acc]
test_acc  = [dt_test_acc, bag_test_acc]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, train_acc, width, label='Train Accuracy')
ax.bar(x + width/2, test_acc,  width, label='Test Accuracy')

ax.set_title('Accuracy Comparison (Regularized)')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# STEP 9 – Confusion Matrix
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay(confusion_matrix(y_test, dt_pred)).plot(ax=axes[0])
axes[0].set_title('Decision Tree')

ConfusionMatrixDisplay(confusion_matrix(y_test, bag_pred)).plot(ax=axes[1])
axes[1].set_title('Bagging Classifier')

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# STEP 10 – ROC Curve
# ─────────────────────────────────────────
dt_probs  = dt_model.predict_proba(X_test_scaled)[:, 1]
bag_probs = bag_model.predict_proba(X_test_scaled)[:, 1]

dt_fpr,  dt_tpr,  _ = roc_curve(y_test, dt_probs)
bag_fpr, bag_tpr, _ = roc_curve(y_test, bag_probs)

dt_auc  = auc(dt_fpr,  dt_tpr)
bag_auc = auc(bag_fpr, bag_tpr)

plt.figure(figsize=(7, 5))
plt.plot(dt_fpr,  dt_tpr,  label=f'Decision Tree (AUC = {dt_auc:.2f})')
plt.plot(bag_fpr, bag_tpr, label=f'Bagging (AUC = {bag_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# STEP 11 – Feature Importance (FIXED)
# ─────────────────────────────────────────
features = list(X.columns)

dt_importance = dt_model.feature_importances_

bag_importance = np.mean(
    [tree.feature_importances_ for tree in bag_model.estimators_],
    axis=0
)

# Ensure same length
min_len = min(len(features), len(dt_importance), len(bag_importance))

features = features[:min_len]
dt_importance = dt_importance[:min_len]
bag_importance = bag_importance[:min_len]

importance_df = pd.DataFrame({
    'Feature': features,
    'Decision Tree': dt_importance,
    'Bagging': bag_importance
})

importance_df = importance_df.sort_values(by='Bagging', ascending=False)

print(importance_df)

# Plot
x = np.arange(len(importance_df))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - width/2, importance_df['Decision Tree'], width, label='Decision Tree')
ax.bar(x + width/2, importance_df['Bagging'], width, label='Bagging')

ax.set_title('Feature Importance Comparison')
ax.set_ylabel('Importance Score')
ax.set_xticks(x)
ax.set_xticklabels(importance_df['Feature'], rotation=30)

ax.legend()
plt.tight_layout()
plt.show()
