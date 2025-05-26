import data_cleaning as dc
import KNN 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

title = "Starships"
artist = "Nicki Minaj"
print("Szukam podobnych utworów do:", title, "wykonawcy:", artist)

# # Znajdź wiersz odpowiadający tytułowi i wykonawcy
# song_row = dc.df[(dc.df['track_name'] == title) & (dc.df['artist_name'] == artist)]

# # Usuń wiersze z brakami w cechach (tak samo jak w X_knn)
# song_row = song_row[KNN.features_knn].dropna()

# if song_row.empty:
#     print("Nie znaleziono piosenki lub braki danych.")
# else:
#     distances, indices = KNN.knn.kneighbors([song_row.iloc[0]])
#     print(dc.df.iloc[indices[0]][['track_name', 'artist_name']])


# Znajdź wiersz w df
song_row_full = dc.df[(dc.df['track_name'] == title) & (dc.df['artist_name'] == artist)]

if song_row_full.empty:
    print("❌ Nie znaleziono piosenki.")
else:
    song_features = song_row_full[KNN.features_knn].dropna()

    if song_features.empty:
        print("❌ Piosenka znaleziona, ale brakuje danych w cechach.")
    else:
        # KNN
        distances, indices = KNN.knn.kneighbors(song_features)
        print("🔎 Podobne utwory:")
        print(dc.df.iloc[indices[0]][['track_name', 'artist_name']])
        
        # Dodatkowe info
        print(f"\n📏 Średni dystans do 5 sąsiadów: {KNN.avg_distance:.4f}")
        # print(f"🧠 Silhouette Score zbioru: {KNN.sil_score:.4f}")
