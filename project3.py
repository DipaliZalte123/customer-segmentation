import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_customer.csv')
print(df.head())

# Elbow Method
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia.append(km.inertia_)

plt.figure()
plt.plot(range(1,11), inertia, 'o-')
plt.title('Elbow Method')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.savefig('elbow.png')
print("Elbow saved!")

# Final Clustering K=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

# Plot Clusters
plt.figure(figsize=(8,6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='rainbow', s=100)
plt.title('5 Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.savefig('clusters.png')
print("Clusters saved!")

# Show insights
print(df.groupby('Cluster').mean(numeric_only=True))
print("Project 3 Done!")