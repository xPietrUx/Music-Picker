import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def prepare_data():
    # Wczytanie danych
    cleaned_path = 'data/spotify_data_cleaned.csv'
    raw_path = 'data/spotify_data.csv'
    if os.path.exists(cleaned_path):
        return pd.read_csv(cleaned_path)
    df = pd.read_csv(raw_path)
    
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
    df.to_csv('data/spotify_data_cleaned.csv', index=False)
    return df

df = prepare_data()