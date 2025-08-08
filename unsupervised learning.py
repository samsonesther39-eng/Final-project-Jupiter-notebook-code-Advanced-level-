# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Scale the data using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

# Determine the optimal number of clusters (K) using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-means clustering with the chosen number of clusters (K=3)
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42)
kmeans.fit(scaled_features)
labels = kmeans.labels_

# Evaluate the clustering using silhouette score
silhouette = silhouette_score(scaled_features, labels)
print("Silhouette Score:", silhouette)

# Visualize the clusters using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels)
plt.title("K-means Clustering")
plt.show()