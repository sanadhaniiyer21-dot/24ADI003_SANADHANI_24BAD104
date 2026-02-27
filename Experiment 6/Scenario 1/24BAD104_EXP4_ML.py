
print("EXP: 4")
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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris


# 1) SMS Spam 
# Update the path to where spam.csv is located on your system
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep required columns only
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# Text preprocessing
df["text"] = df["text"].str.lower()
df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-z ]", "", x))

# Encode labels (ham=0, spam=1)
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Model training
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("SMS Spam - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 2) Iris Classification (Gaussian Naive Bayes)


# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Metrics
acc_iris = accuracy_score(y_test, pred)
report = classification_report(y_test, pred)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True)
plt.title("Iris Dataset - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
