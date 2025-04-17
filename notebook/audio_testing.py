# in this script, we will test and explore the zipped audio dataset,
# with the goal of creating our spectrograms without decompressing everything.
# possibly, a pre filtering having our target dataset with valid tracks' ids would be very nice to save some extra space

# on hold untill i gathered the feature dataset!
#%% explore mp3s zip
import os
import zipfile
import librosa
import io
import sys
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root
from scripts.processing_functions import compute_mel_spectrogram, compute_stft_spectrogram
from scripts.plotting_functions import make_stft_spectrogram_plot, make_mel_spectrogram_plot
# %%
zip_path = os.path.join('..', 'data', 'raw', 'fma_medium.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List all files in the archive
    track_list = [name for name in zip_ref.namelist() if "mp3" in name]
    print(track_list)
    for track in track_list:
        # Read a specific file (e.g., 'data/track1.mp3') as bytes
        with zip_ref.open(track) as file_in_zip:
            audio_bytes = file_in_zip.read()
        print(f"processing track {track}")
        # Process the file (e.g., convert to spectrogram)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=44100)
        # insert bass_band filtering
        stft_spectrogram = compute_stft_spectrogram(y=y)
        stft_spectrogram_plot = make_stft_spectrogram_plot(spectrogram=stft_spectrogram, sr=sr)
        stft_spectrogram_plot.show()
    #mel_spectrogram = compute_mel_spectrogram(y=y)
    #mel_spectrogram_plot = make_mel_spectrogram_plot(spectrogram=mel_spectrogram, sr=sr)
    #mel_spectrogram_plot.show()

# %%
