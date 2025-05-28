import recomender_system.data_cleaning as dc
from sklearn.neighbors import (
    NearestNeighbors,
)
import numpy as np
import os
import pickle

# Definiowanie listy cech używanych przez model KNN.
features_knn = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "loudness",
    "tempo",
]
# Przygotowanie danych treningowych X_knn: wybranie zdefiniowanych cech i usunięcie wierszy z brakującymi wartościami.
X_knn = dc.df[features_knn].dropna()


# Definiowanie funkcji do wczytywania lub trenowania modelu KNN.
def load_knn_model(n_neighbors_param):  # Dodano parametr n_neighbors_param
    # Tworzenie dynamicznej ścieżki do pliku modelu na podstawie liczby sąsiadów.
    model_path = f"data/knn_model_k{n_neighbors_param}.pkl"
    # Sprawdzanie, czy plik modelu już istnieje.
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            knn_loaded = pickle.load(f)
        # Sprawdzanie, czy wczytany model ma atrybut n_neighbors i czy jego wartość zgadza się z oczekiwaną liczbą sąsiadów.
        if (
            hasattr(knn_loaded, "n_neighbors")
            and knn_loaded.n_neighbors == n_neighbors_param
        ):
            print(f"✅ Wczytywanie modelu KNN (k={n_neighbors_param}) z pliku.")
            return knn_loaded
        else:
            print(
                f"⚠️  Model dla k={n_neighbors_param} niezgodny lub brak, trenuję model od nowa..."
            )
    # Trenowanie modelu od nowa, jeśli nie istnieje lub parametry się nie zgadzają.
    # Tworzenie nowego obiektu NearestNeighbors z podaną liczbą sąsiadów i automatycznym wyborem algorytmu.
    knn_new = NearestNeighbors(n_neighbors=n_neighbors_param, algorithm="auto")
    # Trenowanie nowego modelu KNN na danych X_knn.
    knn_new.fit(X_knn)
    # Otwieranie pliku modelu w trybie zapisu binarnego.
    with open(model_path, "wb") as f:
        pickle.dump(knn_new, f)
    print(f"✅ Model KNN (k={n_neighbors_param}) zapisany do pliku.")
    return knn_new
