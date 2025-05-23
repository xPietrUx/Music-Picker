import data_cleaning as dc
from sklearn.neighbors import NearestNeighbors

features_knn = ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence','popularity','loudness', 'tempo']
X_knn = dc.df[features_knn].dropna()

X_knn = dc.df[features_knn].dropna()

knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_knn)
