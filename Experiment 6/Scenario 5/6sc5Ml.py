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
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc

from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────
# STEP 2 – Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv('fraud_smote.csv')

# ─────────────────────────────────────────
# STEP 3 – Features & Target
# ─────────────────────────────────────────
X = df[['Amount', 'Time', 'Feature1', 'Feature2']]
y = df['Fraud']

# ─────────────────────────────────────────
# STEP 4 – Check Class Imbalance
# ─────────────────────────────────────────
print("\nClass Distribution BEFORE SMOTE:")
print(y.value_counts())

plt.figure()
y.value_counts().plot(kind='bar')
plt.title('Class Distribution BEFORE SMOTE')
plt.xticks([0,1], ['Normal (0)', 'Fraud (1)'], rotation=0)
plt.show()

# ─────────────────────────────────────────
# STEP 5 – Train-Test Split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# STEP 6 – Scaling
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# STEP 7 – Model BEFORE SMOTE
# ─────────────────────────────────────────
model_before = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'   
)

model_before.fit(X_train_scaled, y_train)
pred_before = model_before.predict(X_test_scaled)
probs_before = model_before.predict_proba(X_test_scaled)[:, 1]

acc_before = accuracy_score(y_test, pred_before)

print("\nBEFORE SMOTE")
print("Accuracy:", acc_before)
print(classification_report(y_test, pred_before))

# ─────────────────────────────────────────
# STEP 8 – Apply SMOTE
# ─────────────────────────────────────────
smote = SMOTE(
    random_state=42,
    k_neighbors=3   # 🔥 better for small dataset (120 rows)
)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("\nClass Distribution AFTER SMOTE:")
print(pd.Series(y_train_smote).value_counts())

plt.figure()
pd.Series(y_train_smote).value_counts().plot(kind='bar')
plt.title('Class Distribution AFTER SMOTE')
plt.xticks([0,1], ['Normal (0)', 'Fraud (1)'], rotation=0)
plt.show()

# ─────────────────────────────────────────
# STEP 9 – Model AFTER SMOTE
# ─────────────────────────────────────────
model_after = LogisticRegression(max_iter=1000)

model_after.fit(X_train_smote, y_train_smote)
pred_after = model_after.predict(X_test_scaled)
probs_after = model_after.predict_proba(X_test_scaled)[:, 1]

acc_after = accuracy_score(y_test, pred_after)

print("\nAFTER SMOTE")
print("Accuracy:", acc_after)
print(classification_report(y_test, pred_after))

# ─────────────────────────────────────────
# STEP 10 – Precision-Recall Curve (FINAL FIX)
# ─────────────────────────────────────────
precision_before, recall_before, _ = precision_recall_curve(y_test, probs_before)
precision_after, recall_after, _ = precision_recall_curve(y_test, probs_after)

pr_auc_before = auc(recall_before, precision_before)
pr_auc_after  = auc(recall_after, precision_after)

plt.clf()
plt.close('all')

plt.figure()

plt.plot(recall_before, precision_before,
         linestyle='--', linewidth=5,
         label=f'Before SMOTE (AUC = {pr_auc_before:.2f})')

plt.plot(recall_after, precision_after,
         linewidth=2,
         label=f'After SMOTE (AUC = {pr_auc_after:.2f})')

# baseline
baseline = sum(y_test) / len(y_test)
plt.hlines(baseline, 0, 1, linestyles='dashed', label='Baseline')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.legend()
plt.grid(True)

plt.show()
