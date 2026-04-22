# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 2. Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Preview:")
print(df.head())


# 3. Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Choose number of components using AIC and BIC
aic = []
bic = []
n_components = range(1,11)

for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    aic.append(gmm.aic(X_scaled))
    bic.append(gmm.bic(X_scaled))

plt.figure()
plt.plot(n_components, aic, label='AIC')
plt.plot(n_components, bic, label='BIC')
plt.xlabel("Number of Components")
plt.ylabel("Score")
plt.title("AIC/BIC for Optimal Clusters")
plt.legend()
plt.show()


# 6. Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X_scaled)


# 7. Predict cluster probabilities
probabilities = gmm.predict_proba(X_scaled)

print("\nCluster Probabilities (first 5 rows):")
print(probabilities[:5])


# 8. Assign clusters based on highest probability
clusters = gmm.predict(X_scaled)
df['GMM Cluster'] = clusters


# 9. Evaluate clustering

# Log Likelihood
log_likelihood = gmm.score(X_scaled)
print("\nLog Likelihood:", log_likelihood)

# Silhouette Score
sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", sil_score)

# AIC and BIC
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))


# 10. Visualization of GMM clusters
plt.figure()
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='viridis')
plt.title("GMM Customer Clusters")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()


# 11. K-Means for comparison
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

plt.figure()
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans_clusters, cmap='plasma')
plt.title("K-Means Clusters")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()
