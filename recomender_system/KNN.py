import recomender_system.data_cleaning as dc
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import pickle

k = 3
n = 500
features_knn = ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence','loudness', 'tempo']
X_knn = dc.df[features_knn].dropna()

model_path = 'data/knn_model.pkl'

def load_knn_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            knn_loaded = pickle.load(f)
        # Sprawdź czy liczba sąsiadów się zgadza
        if hasattr(knn_loaded, 'n_neighbors') and knn_loaded.n_neighbors == k:
            print("✅ Wczytywanie modelu KNN z pliku.")
            return knn_loaded
        else:
            print("⚠️  Zmieniono k, trenuję model od nowa...")
    # Trenuj model od nowa
    knn_new = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_new.fit(X_knn)
    with open(model_path, 'wb') as f:
        pickle.dump(knn_new, f)
    print("✅ Model KNN zapisany do pliku.")
    return knn_new

knn = load_knn_model()

# Średni dystans rekomendacji

print(f"Obliczanie średniego dystansu na próbce {n}...")
sample_size = min(n, len(X_knn))  # np. 500 lub mniej, jeśli danych jest mniej
sample = X_knn.sample(sample_size, random_state=42)
distances_all, indices_all = knn.kneighbors(sample)
avg_distance = np.mean(distances_all[:, 1:])
print("Model gotowy.")