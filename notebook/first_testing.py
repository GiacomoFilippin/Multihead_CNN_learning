import zipfile
import io

# Open the ZIP file
with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    # List all files in the archive
    print(zip_ref.namelist())
    
    # Read a specific file (e.g., 'data/track1.mp3') as bytes
    with zip_ref.open('data/track1.mp3') as file_in_zip:
        audio_bytes = file_in_zip.read()
    
    # Process the file (e.g., convert to spectrogram)
    import librosa
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=44100)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)