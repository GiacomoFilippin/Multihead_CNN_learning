import pandas as pd
import os
import sys
project_root = os.path.dirname(os.path.abspath(''))
sys.path.append(project_root)  # Ora Python trova i moduli nella root
def load_features(folder=project_root+"\\data\\processed\\fma_metadata"):
    # Carica i metadati
    tracks = pd.read_csv(f"{folder}\\tracks.csv", index_col=0, header=[0, 1])
    genres = pd.read_csv(f"{folder}\\genres.csv", index_col=0)
    features = pd.read_csv(f"{folder}\\features.csv", index_col=0)
    echonest = pd.read_csv(f"{folder}\\echonest.csv", index_col=0)
    
    return tracks, genres, features, echonest