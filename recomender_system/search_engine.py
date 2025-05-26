import recomender_system.data_cleaning as dc
import recomender_system.KNN as KNN

def find_similar_songs(title, artist):
    print(f"Szukam podobnych utworów do: {title} wykonawcy: {artist}")

    # Znajdź wiersz w df
    song_row_full = dc.df[(dc.df['track_name'] == title) & (dc.df['artist_name'] == artist)]

    if song_row_full.empty:
        print("❌ Nie znaleziono piosenki.")
        return

    song_features = song_row_full[KNN.features_knn].dropna()

    if song_features.empty:
        print("❌ Piosenka znaleziona, ale brakuje danych w cechach.")
        return

    try:
        # Przekaż dane jako DataFrame, nie numpy array
        distances, indices = KNN.knn.kneighbors(song_features)
        print("🔎 Podobne utwory:")
        print(dc.df.iloc[indices[0]][['track_name', 'artist_name']])
        print(f"\n📏 Średni dystans do {KNN.k} sąsiadów: {KNN.avg_distance:.4f}")
    except Exception as e:
        print(f"Błąd podczas wyszukiwania podobnych utworów: {e}")

if __name__ == "__main__":
    find_similar_songs("Summertime Sadness", "Lana Del Rey")
