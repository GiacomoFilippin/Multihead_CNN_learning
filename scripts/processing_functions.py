import utils

def load_features(folder="data/processed/fma_metadata/tracks.csv"):
    tracks = utils.load(folder)
    genres = utils.load(folder)
    features = utils.load(folder)
    echonest = utils.load(folder)
    return tracks, genres, features, echonest