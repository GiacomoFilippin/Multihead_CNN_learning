# in this script, we will test and explore the feature space, 
# with the goal of deciding which features are suited for our multitask study 
# %%
import zipfile
import io
import os 
import sys
import numpy as np
# Aggiungi la root del progetto al Python path
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root
from scripts.processing_functions import load_features
# %% Load metadata and features.
tracks, genres, features, echonest = load_features()
print('Echonest features available for {} tracks.'.format(len(echonest)))
# %%
features_clean = features.iloc[3:]
np.testing.assert_array_equal(features_clean.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

tracks_medium = tracks[tracks['set', 'subset'] <= 'medium']

tracks_medium.shape, features_clean.shape, echonest.shape

durations = tracks_medium['track', 'duration']
genres = tracks_medium['track', 'genre_top']