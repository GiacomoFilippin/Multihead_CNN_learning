#%%
import zipfile
import io
import os 
import sys
from scripts.processing_functions import load_features
# Aggiungi la root del progetto al Python path
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root
# Load metadata and features.
tracks, genres, features, echonest = load_features()
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
