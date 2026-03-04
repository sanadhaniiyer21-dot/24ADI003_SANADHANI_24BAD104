print("Sanadhani -24BAD104")
print("Exp-4")

# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# =========================
# 3. Text Preprocessing
# =========================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_message'] = df['message'].apply(clean_text)

# =========================
# 4. Convert Text to Numerical Features (TF-IDF)
# =========================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])

# =========================
# 5. Encode Target Labels
# =========================
le = LabelEncoder()
y = le.fit_transform(df['label'])   # ham = 0, spam = 1

print("Preprocessing Completed Successfully")
print("Feature Matrix Shape:", X.shape)

# =========================
# 6. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# =========================
# 7. Train Multinomial Naïve Bayes Model
# =========================
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model Training Completed Successfully")

# =========================
# 8. Make Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# 9. Model Evaluation
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance (alpha = 1.0)")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# =========================
# 10. Analyze Misclassified Examples
# =========================
misclassified_indices = np.where(y_test != y_pred)[0]

print("\nNumber of Misclassified Messages:", len(misclassified_indices))

for i in misclassified_indices[:5]:
    print("\nMessage:")
    print(df.iloc[i]['message'])
    print("Actual   :", le.inverse_transform([y_test[i]])[0])
    print("Predicted:", le.inverse_transform([y_pred[i]])[0])

# =========================
# 11. Apply Laplace Smoothing (alpha = 0.5)
# =========================
model_smooth = MultinomialNB(alpha=0.5)
model_smooth.fit(X_train, y_train)

y_pred_smooth = model_smooth.predict(X_test)

print("\nModel Performance (alpha = 0.5)")
print("Accuracy :", round(accuracy_score(y_test, y_pred_smooth), 4))
print("Precision:", round(precision_score(y_test, y_pred_smooth), 4))
print("Recall   :", round(recall_score(y_test, y_pred_smooth), 4))
print("F1 Score :", round(f1_score(y_test, y_pred_smooth), 4))

# =========================
# 12. Confusion Matrix
# =========================
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# =========================
# Confusion Matrix (alpha = 0.5)
# =========================
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_smooth = confusion_matrix(y_test, y_pred_smooth)

plt.figure()
sns.heatmap(cm_smooth, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix (alpha = 0.5)")
plt.show()

# =========================
# 13. Feature Importance (Top Spam Words)
# =========================

feature_names = vectorizer.get_feature_names_out()

# Get log probabilities of spam class (class = 1)
spam_log_probs = model.feature_log_prob_[1]

# Get top 15 words with highest probability in spam
top_spam_indices = np.argsort(spam_log_probs)[-15:]

top_spam_words = feature_names[top_spam_indices]
top_spam_values = spam_log_probs[top_spam_indices]

plt.figure()
plt.barh(top_spam_words, top_spam_values)
plt.xlabel("Log Probability")
plt.title("Top Words Influencing Spam Classification")
plt.show()

# =========================
# 14. Word Frequency Comparison (Spam vs Ham)
# =========================

# Separate spam and ham messages
spam_messages = df[df['label'] == 'spam']['clean_message']
ham_messages = df[df['label'] == 'ham']['clean_message']

# Vectorize separately
spam_counts = vectorizer.transform(spam_messages).toarray().sum(axis=0)
ham_counts = vectorizer.transform(ham_messages).toarray().sum(axis=0)

# Get top 10 common words in spam and ham
top_spam_freq_indices = np.argsort(spam_counts)[-10:]
top_ham_freq_indices = np.argsort(ham_counts)[-10:]

plt.figure()
plt.bar(feature_names[top_spam_freq_indices], spam_counts[top_spam_freq_indices])
plt.xticks(rotation=45)
plt.title("Top Word Frequencies in Spam")
plt.show()

plt.figure()
plt.bar(feature_names[top_ham_freq_indices], ham_counts[top_ham_freq_indices])
plt.xticks(rotation=45)
plt.title("Top Word Frequencies in Ham")
plt.show()

