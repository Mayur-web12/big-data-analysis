# Import libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.cluster import KMeans

# Step 1: Load dataset

iris = load_iris()

X = iris.data

y = iris.target

# Step 2: Apply K-means clustering

kmeans = KMeans(
    n_clusters=3,
    random_state=123,
    n_init=25
)

clusters = kmeans.fit_predict(X)

# Step 3: Add cluster labels

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df['cluster'] = clusters

df['species'] = y

# Step 4: Evaluate clusters

print(pd.crosstab(
    df['species'],
    df['cluster']
))

# Step 5: Plot clusters

colors = ['red', 'blue', 'green']

plt.figure(figsize=(10,6))

for i in range(3):

    plt.scatter(
        df[df['cluster']==i]['sepal length (cm)'],
        df[df['cluster']==i]['sepal width (cm)'],
        c=colors[i],
        label=f'Cluster {i}',
        s=80
    )

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")

plt.title("K-means Clustering of Iris Data")

plt.legend()

plt.show()
