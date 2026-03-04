print("Sanadhani -24BAD104")
print("Exp-4")
# =========================
# 1. Import Libraries
# =========================
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 2. Load Dataset
# =========================
iris = load_iris()

X = iris.data          # Features
y = iris.target        # Target labels

feature_names = iris.feature_names
target_names = iris.target_names

print("Feature Names:", feature_names)
print("Target Classes:", target_names)
print("Dataset Shape:", X.shape)

# =========================
# 3. Data Inspection
# =========================

# Convert to DataFrame for easier inspection
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df['species'].value_counts())

# =========================
# 4. Feature Scaling
# =========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed")

# =========================
# Train-Test Split
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# =========================
# 6. Train Gaussian Naïve Bayes Model
# =========================
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("Gaussian Naïve Bayes Model Trained Successfully")

# =========================
# 7. Make Predictions
# =========================
y_pred = gnb.predict(X_test)

print("Prediction Completed Successfully")

# Optional: View first 10 predictions
print("\nFirst 10 Predictions:")
for i in range(10):
    print("Predicted:", y_pred[i], "Actual:", y_test[i])
    
# =========================
# 8. Model Evaluation
# =========================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Performance")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# =========================
# 10. Analyze Misclassified Examples
# =========================
import numpy as np

misclassified_indices = np.where(y_test != y_pred)[0]

print("\nNumber of Misclassified Samples:", len(misclassified_indices))

for i in misclassified_indices:
    print("\nFeatures:", X_test[i])
    print("Actual   :", iris.target_names[y_test[i]])
    print("Predicted:", iris.target_names[y_pred[i]])
    
    
# =========================
# Decision Boundary Plot
# =========================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Use only 2 features
X_two = df[['petal length (cm)', 'petal width (cm)']].values
y_two = df['species'].values

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_two, y_two, test_size=0.2, random_state=42
)

model2 = GaussianNB()
model2.fit(X_train2, y_train2)

# Create meshgrid
x_min, x_max = X_two[:, 0].min() - 1, X_two[:, 0].max() + 1
y_min, y_max = X_two[:, 1].min() - 1, X_two[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_two[:, 0], X_two[:, 1], c=y_two, edgecolor='k')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Decision Boundary - Gaussian Naïve Bayes')
plt.show()

# =========================
# Confusion Matrix
# =========================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Gaussian NB")
plt.show()

# =========================
# Probability Distribution Plots
# =========================
import seaborn as sns

features = df.columns[:-1]  # exclude species column

for feature in features:
    plt.figure()
    for species in df['species'].unique():
        sns.kdeplot(
            df[df['species'] == species][feature],
            label=f"Class {species}"
        )
    plt.title(f"Distribution of {feature}")
    plt.legend()
    plt.show()
