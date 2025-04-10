# in this script, we will test and explore the zipped audio dataset,
# with the goal of creating our spectrograms without decompressing everything.
# possibly, a pre filtering having our target dataset with valid tracks' ids would be very nice to save some extra space

# on hold untill i gathered the feature dataset!
#%% explore mp3s zip
import os
import zipfile
# %%
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