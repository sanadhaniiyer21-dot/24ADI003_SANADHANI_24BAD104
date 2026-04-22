
print("24BAD104-Sanadhani")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# 2. Load the dataset
df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# 3. Data preprocessing
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# 4. Select relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Elbow Method to determine optimal K
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()


# 6. Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


# 7. Assign cluster labels
df['Cluster'] = clusters

print("\nDataset with Cluster Labels:")
print(df.head())


# 8. Visualize clusters
plt.figure()
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='viridis')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=200,
            c='red',
            marker='X')

plt.title("Customer Segments")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()


# 9. Evaluation Metrics

# Inertia
print("\nInertia:", kmeans.inertia_)

# Silhouette Score
sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", sil_score)

