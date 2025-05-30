import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tabulate import tabulate

try:
    df = pd.read_csv("data/spotify_data.csv")
except FileNotFoundError:
    print(
        "Błąd: Plik CSV ze zbioru danych nie został znaleziony. Upewnij się, że plik istnieje w podanej lokalizacji."
    )
    exit()

feature_cols = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

df_cleaned = df.copy()
# Upewnij się, że nazwy kolumn 'track_name', 'artist_name' są zgodne z Twoim plikiem CSV
df_cleaned.dropna(subset=["track_name", "artist_name"], inplace=True)

missing_features = [col for col in feature_cols if col not in df_cleaned.columns]
if missing_features:
    print(
        f"Błąd: Brakuje następujących kolumn cech w pliku CSV: {', '.join(missing_features)}"
    )
    exit()

df_cleaned.dropna(subset=feature_cols, inplace=True)

if df_cleaned.empty:
    print(
        "Błąd: DataFrame jest pusty po usunięciu brakujących wartości. Sprawdź dane wejściowe."
    )
    exit()

scaler = StandardScaler()
df_cleaned[feature_cols] = scaler.fit_transform(df_cleaned[feature_cols])

model = NearestNeighbors(n_neighbors=10, algorithm="ball_tree")
model.fit(df_cleaned[feature_cols])


def find_similar_songs_with_details(track_name, artist_name, n_neighbors=5):
    # Upewnij się, że nazwy kolumn 'track_name', 'artist_name' są zgodne z Twoim plikiem CSV
    song_query = df_cleaned[
        (df_cleaned["track_name"].str.lower() == track_name.lower())
        & (df_cleaned["artist_name"].str.lower() == artist_name.lower())
    ]

    if song_query.empty:
        print(
            f"Piosenka '{track_name}' wykonawcy '{artist_name}' nie została znaleziona w zbiorze danych."
        )
        similar_titles = df_cleaned[
            df_cleaned["track_name"].str.contains(track_name, case=False, na=False)
        ]
        if not similar_titles.empty:
            print("\nCzy chodziło Ci o którąś z tych piosenek?")
            for i, (idx, r) in enumerate(
                similar_titles[["track_name", "artist_name"]]
                .drop_duplicates()
                .head()
                .iterrows()
            ):
                print(f"{i+1}. {r['track_name']} by {r['artist_name']}")
        return

    # Pobierz oryginalne dane piosenki wyszukiwanej z df
    query_song_original_index = song_query.index[0]
    query_song_original_data = df.loc[query_song_original_index]

    # Pobierz przeskalowane cechy piosenki wyszukiwanej do modelu KNN
    song_features_scaled = song_query[feature_cols].iloc[[0]].values

    distances, indices = model.kneighbors(
        song_features_scaled, n_neighbors=n_neighbors + 1
    )

    similar_songs_indices = []
    similar_songs_distances = []

    for i in range(len(indices[0])):  # Iteruj po wszystkich znalezionych sąsiadach
        # Indeks znalezionego sąsiada w df_cleaned
        neighbor_df_cleaned_index = df_cleaned.index[indices[0][i]]

        # Sprawdź, czy znaleziony sąsiad to nie ta sama piosenka co zapytanie
        if neighbor_df_cleaned_index != query_song_original_index:
            similar_songs_indices.append(
                neighbor_df_cleaned_index
            )  # Zapisz indeks z df_cleaned (lub oryginalnego df, jeśli są spójne)
            similar_songs_distances.append(distances[0][i])

        if (
            len(similar_songs_indices) == n_neighbors
        ):  # Zatrzymaj, gdy mamy wystarczająco dużo sąsiadów
            break

    if not similar_songs_indices:
        print(
            f"Nie znaleziono innych podobnych piosenek dla '{track_name}' wykonawcy '{artist_name}'."
        )
        return

    # Przygotowanie danych do jednej tabeli
    table_data = []
    headers = ["Piosenka", "Artysta", "Dystans"] + feature_cols

    # Wiersz dla piosenki wyszukiwanej (oryginalne wartości cech)
    query_row_display = [
        query_song_original_data["track_name"],
        query_song_original_data["artist_name"],
        "N/A (Piosenka Bazowa)",
    ]
    for feature in feature_cols:
        val = query_song_original_data[feature]
        query_row_display.append(
            f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
        )
    table_data.append(query_row_display)

    # Wiersze dla piosenek podobnych (oryginalne wartości cech)
    for i, similar_song_df_index in enumerate(similar_songs_indices):
        # Pobierz oryginalne dane podobnej piosenki z df
        row_original_similar = df.loc[similar_song_df_index]

        similar_song_row_display = [
            row_original_similar["track_name"],
            row_original_similar["artist_name"],
            f"{similar_songs_distances[i]:.4f}",
        ]
        for feature in feature_cols:
            val = row_original_similar[feature]
            similar_song_row_display.append(
                f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
            )
        table_data.append(similar_song_row_display)

    # Wyświetlenie jednej tabeli
    print(f"\nPorównanie piosenki '{track_name}' z podobnymi utworami:")
    print(
        tabulate(
            table_data,
            headers=headers,
            tablefmt="grid",
            stralign="left",
            numalign="right",
        )
    )
    # Koniec logiki związanej z tabelą

    # Wyświetlenie średniego dystansu (osobno, po tabeli)
    if len(similar_songs_distances) > 0:
        avg_distance = sum(similar_songs_distances) / len(similar_songs_distances)
        print(f"\nŚredni dystans do podobnych piosenek: {avg_distance:.4f}")


if __name__ == "__main__":
    if "df_cleaned" in globals() and not df_cleaned.empty:
        find_similar_songs_with_details(
            "Timber (feat. Ke$ha)", "Pitbull", n_neighbors=3
        )
    else:
        print(
            "Nie można uruchomić wyszukiwania, ponieważ dane nie zostały poprawnie załadowane lub przetworzone."
        )
