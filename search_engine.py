import data_cleaning as dc
import KNN 

title = "Starships"
artist = "Nicki Minaj"

# Znajdź wiersz odpowiadający tytułowi i wykonawcy
song_row = dc.df[(dc.df['track_name'] == title) & (dc.df['artist_name'] == artist)]

# Usuń wiersze z brakami w cechach (tak samo jak w X_knn)
song_row = song_row[KNN.features_knn].dropna()

if song_row.empty:
    print("Nie znaleziono piosenki lub braki danych.")
else:
    distances, indices = KNN.knn.kneighbors([song_row.iloc[0]])
    print(dc.df.iloc[indices[0]][['track_name', 'artist_name']])