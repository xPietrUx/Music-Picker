import data_cleaning as dc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import numpy as np

k = 6 

features_knn = ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence','loudness', 'tempo']
X_knn = dc.df[features_knn].dropna()

knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
knn.fit(X_knn)

# Odległości
distances_all, indices_all = knn.kneighbors(X_knn)
avg_distance = np.mean(distances_all[:, 1:])

# # Silhouette score
# labels = np.zeros(X_knn.shape[0], dtype=int)
# for i, neighbors in enumerate(indices_all[:, 1:]):
#     labels[neighbors] = i

# if len(set(labels)) > 1:
#     sil_score = silhouette_score(X_knn, labels)
# else:
#     sil_score = -1


globals()['avg_distance'] = avg_distance
# globals()['sil_score'] = sil_score