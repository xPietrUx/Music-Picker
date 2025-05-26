import recomender_system.data_cleaning as dc  # Importowanie modułu do czyszczenia danych
import recomender_system.KNN as KNN  # Importowanie modułu KNN


def find_similar_songs(
    title="Starships", artist="Nicki Minaj", numberOfSongs=10
):  # Definiowanie funkcji do wyszukiwania podobnych utworów
    print(  # Wyświetlanie informacji o rozpoczęciu wyszukiwania
        f"🔦 Szukam podobnych utworów do: {title} wykonawcy: {artist} (liczba rekomendacji: {numberOfSongs})"
    )

    # Lista zwracająca podobne utwory
    recommended_songs = []
    # Średni dystans
    avg_dist_for_query = None

    # Znajdź wiersz w df
    song_row_full = (
        dc.df[  # Wyszukiwanie utworu w DataFrame na podstawie tytułu i artysty
            (dc.df["track_name"] == title) & (dc.df["artist_name"] == artist)
        ]
    )

    if song_row_full.empty:  # Sprawdzanie, czy utwór został znaleziony
        print(
            "❌ Nie znaleziono piosenki."
        )  # Wyświetlanie komunikatu o nieznalezieniu utworu
        return recommended_songs, avg_dist_for_query

    song_features = song_row_full[
        KNN.features_knn
    ].dropna()  # Pobieranie cech utworu i usuwanie brakujących wartości

    if song_features.empty:  # Sprawdzanie, czy istnieją cechy dla znalezionego utworu
        print(
            "❌ Piosenka znaleziona, ale brakuje danych w cechach."
        )  # Wyświetlanie komunikatu o braku danych w cechach
        return recommended_songs, avg_dist_for_query

    try:
        # Ustalanie liczby sąsiadów dla modelu KNN
        k_for_model = numberOfSongs + 1

        # Załaduj/pobierz model KNN z określoną wartością k
        knn_model = KNN.load_knn_model(
            n_neighbors_param=k_for_model
        )  # Ładowanie lub pobieranie modelu KNN

        # Przekaż dane jako DataFrame, nie numpy array
        distances, indices = knn_model.kneighbors(  # Wyszukiwanie najbliższych sąsiadów
            song_features
        )  # Użycie załadowanego modelu knn_model

        # Pobierz rekomendowane utwory
        similar_songs_df = dc.df.iloc[
            indices[0]
        ]  # Pobieranie danych o podobnych utworach z DataFrame

        # Iteruj po znalezionych utworach i dodaj je do listy
        for (
            index,
            row,
        ) in (
            similar_songs_df.iterrows()
        ):  # Iterowanie po znalezionych podobnych utworach
            # Pomiń utwór wejściowy, jeśli jest w rekomendacjach
            if (
                row["track_name"] == title and row["artist_name"] == artist
            ):  # Sprawdzanie, czy utwór nie jest utworem wejściowym
                continue  # Pomijanie utworu wejściowego
            recommended_songs.append(  # Dodawanie utworu do listy rekomendacji
                {"track_name": row["track_name"], "artist_name": row["artist_name"]}
            )
            # Dodawanie do ilości rządanej przez użytkownika
            if (
                len(recommended_songs) >= numberOfSongs
            ):  # Sprawdzanie, czy osiągnięto żądaną liczbę rekomendacji
                break  # Przerywanie pętli, jeśli osiągnięto limit

        print("🔎 Podobne utwory:")  # Wyświetlanie nagłówka dla listy podobnych utworów
        for song in recommended_songs:  # Iterowanie po liście rekomendowanych utworów
            print(
                f"- {song['track_name']} by {song['artist_name']}"
            )  # Wyświetlanie każdego rekomendowanego utworu
        if distances.size > 0:
            avg_dist_for_query = distances[0].mean()
            print(
                f"\n📏 Średni dystans do {numberOfSongs} potencjalnych rekomendacji (przed odfiltrowaniem utworu wejściowego): {avg_dist_for_query:.4f}"
            )

    except Exception as e:  # Obsługiwanie wyjątków
        print(
            f"⚠️ Błąd podczas wyszukiwania podobnych utworów: {e}"
        )  # Wyświetlanie komunikatu o błędzie

    return recommended_songs, avg_dist_for_query
