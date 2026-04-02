# ─────────────────────────────────────────
# STEP 1 – Import Libraries
# ─────────────────────────────────────────
import matplotlib.pyplot as plt
plt.close('all')
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

# ─────────────────────────────────────────
# STEP 2 – Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv('churn_boosting.csv')

# ─────────────────────────────────────────
# STEP 3 – Encode Categorical Columns
# ─────────────────────────────────────────
le = LabelEncoder()

df['ContractType']    = le.fit_transform(df['ContractType'].astype(str))
df['InternetService'] = le.fit_transform(df['InternetService'].astype(str))

if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ─────────────────────────────────────────
# STEP 4 – Select Features
# ─────────────────────────────────────────
X = df[['Tenure', 'MonthlyCharges', 'ContractType']]
y = df['Churn']

# ─────────────────────────────────────────
# STEP 5 – Train-Test Split (FIXED)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ─────────────────────────────────────────
# STEP 6 – Scaling
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# STEP 7 – AdaBoost (Regularized)
# ─────────────────────────────────────────
base_tree = DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=5
)

ada_model = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

ada_model.fit(X_train_scaled, y_train)
ada_pred = ada_model.predict(X_test_scaled)

# ─────────────────────────────────────────
# STEP 8 – Gradient Boosting (Regularized)
# ─────────────────────────────────────────
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)

# ─────────────────────────────────────────
# STEP 9 – Accuracy
# ─────────────────────────────────────────
ada_train_acc = accuracy_score(y_train, ada_model.predict(X_train_scaled))
ada_test_acc  = accuracy_score(y_test,  ada_pred)

gb_train_acc  = accuracy_score(y_train, gb_model.predict(X_train_scaled))
gb_test_acc   = accuracy_score(y_test,  gb_pred)

print("\nAdaBoost          - Train Accuracy:", ada_train_acc)
print("AdaBoost          - Test Accuracy :", ada_test_acc)
print("Gradient Boosting - Train Accuracy:", gb_train_acc)
print("Gradient Boosting - Test Accuracy :", gb_test_acc)

# ─────────────────────────────────────────
# STEP 10 – ROC Curve (FIXED)
# ─────────────────────────────────────────
plt.clf()
plt.close('all')

# Correct order
ada_probs = ada_model.predict_proba(X_test_scaled)[:, 1]
gb_probs  = gb_model.predict_proba(X_test_scaled)[:, 1]

print("\nClass distribution in test set:")
print(y_test.value_counts())

print("\nUnique predicted probabilities:")
print("AdaBoost:", np.unique(ada_probs))
print("Gradient Boosting:", np.unique(gb_probs))

# Safety check
if len(np.unique(y_test)) < 2:
    print("ERROR: Only one class in test set. ROC cannot be computed.")
else:
    ada_fpr, ada_tpr, _ = roc_curve(y_test, ada_probs)
    gb_fpr,  gb_tpr,  _ = roc_curve(y_test, gb_probs)

    ada_auc = auc(ada_fpr, ada_tpr)
    gb_auc  = auc(gb_fpr,  gb_tpr)

    plt.figure()
    plt.plot(ada_fpr, ada_tpr, linewidth=6,
         label=f'AdaBoost (AUC = {ada_auc:.2f})')
    plt.plot(gb_fpr,  gb_tpr,  label=f'Gradient Boosting (AUC = {gb_auc:.2f})')
    plt.plot([0, 1], [0, 1], '--', label='Random Guess')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    plt.legend()
    plt.show()

# ─────────────────────────────────────────
# STEP 11 – Feature Importance
# ─────────────────────────────────────────
features = list(X.columns)

ada_importance = ada_model.feature_importances_
gb_importance  = gb_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'AdaBoost': ada_importance,
    'Gradient Boosting': gb_importance
})

importance_df = importance_df.sort_values(by='Gradient Boosting', ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# Plot
x = np.arange(len(importance_df))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, importance_df['AdaBoost'], width, label='AdaBoost')
plt.bar(x + width/2, importance_df['Gradient Boosting'], width, label='Gradient Boosting')

plt.xticks(x, importance_df['Feature'], rotation=20)
plt.ylabel('Importance Score')
plt.title('Feature Importance Comparison')

plt.legend()
plt.tight_layout()
plt.show()
