import recomender_system.data_cleaning as dc
import recomender_system.KNN as KNN


def find_similar_songs(title, artist):
    print(f"Szukam podobnych utworów do: {title} wykonawcy: {artist}")

    # Lista zwracająca podobne utwory
    recommended_songs = []

    # Znajdź wiersz w df
    song_row_full = dc.df[
        (dc.df["track_name"] == title) & (dc.df["artist_name"] == artist)
    ]

    if song_row_full.empty:
        print("❌ Nie znaleziono piosenki.")
        return recommended_songs

    song_features = song_row_full[KNN.features_knn].dropna()

    if song_features.empty:
        print("❌ Piosenka znaleziona, ale brakuje danych w cechach.")
        return recommended_songs

    try:
        # Przekaż dane jako DataFrame, nie numpy array
        distances, indices = KNN.knn.kneighbors(song_features)

        # Pobierz rekomendowane utwory
        similar_songs_df = dc.df.iloc[indices[0]]

        # Iteruj po znalezionych utworach i dodaj je do listy
        for index, row in similar_songs_df.iterrows():
            # Pomiń utwór wejściowy, jeśli jest w rekomendacjach
            if row["track_name"] == title and row["artist_name"] == artist:
                continue
            recommended_songs.append(
                {"track_name": row["track_name"], "artist_name": row["artist_name"]}
            )

        print("🔎 Podobne utwory:")
        for song in recommended_songs:
            print(f"- {song['track_name']} by {song['artist_name']}")
        print(f"\n📏 Średni dystans do {KNN.k} sąsiadów: {KNN.avg_distance:.4f}")

    except Exception as e:
        print(f"Błąd podczas wyszukiwania podobnych utworów: {e}")

    return recommended_songs


if __name__ == "__main__":
    results = find_similar_songs("Summertime Sadness", "Lana Del Rey")
    print(f"\nZwrócona lista zawiera {len(results)} utworów")
