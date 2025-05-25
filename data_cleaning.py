import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Wczytanie danych
df = pd.read_csv('data/spotify_data.csv')

# Ogólne czyszczenie danych
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Usunięcie kolumn niepotrzebnych do analizy
df.drop(['Unnamed: 0','track_id', 'mode', 'duration_ms', 'time_signature'], axis=1, inplace=True)

# Usunięcie piosenek, które mają popularność od 0 do 15
df = df[~df['popularity'].between(0, 15)]

# Skalowanie wszystkich cech numerycznych
features_to_scale = ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence','loudness', 'tempo', 'popularity']
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])


# print(df.head())
# print(df.shape)