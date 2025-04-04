fma_medium.zip contains raw, 30s mp3 audio tracks.
the idea is to process the various tracks without decompressing the dataset
this way i can extract features without dimensionality issues,
and i can just save each 30s mp3's spectrogram, alongside its targets (yet to be decided)

fma_metadata.zip contains CSVs about features, such as:
'fma_metadata/checksums', 
'fma_metadata/not_found.pickle', 
'fma_metadata/raw_genres.csv', 
'fma_metadata/raw_albums.csv', 
'fma_metadata/raw_artists.csv', 
'fma_metadata/raw_tracks.csv', 
'fma_metadata/tracks.csv', 
'fma_metadata/genres.csv', 
'fma_metadata/raw_echonest.csv', 
'fma_metadata/echonest.csv', 
'fma_metadata/features.csv'
further exploration of which could be the best tergets to test multitask learning has to be made.