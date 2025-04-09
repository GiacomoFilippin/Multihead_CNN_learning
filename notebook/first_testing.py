#%%
import zipfile
import io
import os 
import sys
import numpy as np
# Aggiungi la root del progetto al Python path
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root
from scripts.processing_functions import load_features
# Load metadata and features.
tracks, genres, features, echonest = load_features()
print('Echonest features available for {} tracks.'.format(len(echonest)))

features_clean = features.iloc[3:]
np.testing.assert_array_equal(features_clean.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

tracks_medium = tracks[tracks['set', 'subset'] <= 'medium']

tracks_medium.shape, features_clean.shape, echonest.shape

durations = tracks_medium['track', 'duration']
genres = tracks_medium['track', 'genre_top']
#%% explore mp3s zip
zip_path = os.path.join('..', 'data', 'raw', 'fma_medium.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List all files in the archive
    print(zip_ref.namelist())
    # Read a specific file (e.g., 'data/track1.mp3') as bytes
    with zip_ref.open('fma_medium/README.txt') as file_in_zip:
        audio_bytes = file_in_zip.read()
"""    
    # Process the file (e.g., convert to spectrogram)
    import librosa
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=44100)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)"""
# %%
print(audio_bytes)
# %% explore metadata zips
zip_path = os.path.join('..', 'data', 'raw', 'fma_metadata.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List all files in the archive
    print(zip_ref.namelist())

# %%
