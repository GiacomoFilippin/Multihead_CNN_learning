from pathlib import Path
import io
import zipfile
import tensorflow as tf
import numpy as np
import librosa
import sys

sys.path.insert(0, str(Path(__file__).parent))
from processing_functions import compute_mel_spectrogram, extract_track_id, load_features

def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(spec: np.ndarray,
                      track_id: int,
                      title: str,
                      father_genre: str,
                      sub_genre: str) -> bytes:
    raw = spec.astype(np.float32).tobytes()
    feat = {
        'spectrogram':  _bytes_feature(raw),
        'height':       _int64_feature(spec.shape[0]),
        'width':        _int64_feature(spec.shape[1]),
        'track_id':     _int64_feature(track_id),
        'title':        _bytes_feature(title.encode('utf-8')),
        'father_genre': _bytes_feature(father_genre.encode('utf-8')),
        'sub_genre':    _bytes_feature(sub_genre.encode('utf-8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feat))
    return example.SerializeToString()

def create_tfrecords(output_path: Path,
                     audio_zip_path: Path,
                     sample_rate: int = 44100):
    tracks_df, _, _, _ = load_features()
    print(f"> Loaded metadata for {len(tracks_df)} tracks")

    with zipfile.ZipFile(audio_zip_path, 'r') as zf:
        valid = []
        for fname in zf.namelist():
            base = Path(fname).name
            try:
                tid = int(extract_track_id(base))
            except Exception:
                continue
            if tid in tracks_df.index:
                valid.append(fname)

        print(f"> Found {len(valid)} audio files with metadata")
        writer = tf.io.TFRecordWriter(str(output_path))
        written = 0

        for idx, arcname in enumerate(valid, start=1):
            base = Path(arcname).name
            tid  = int(extract_track_id(base))
            print(f"[{idx}/{len(valid)}] {base}", end=' → ')

            try:
                raw_data = zf.read(arcname)
                y, _ = librosa.load(io.BytesIO(raw_data), sr=sample_rate)
                spec = compute_mel_spectrogram(y, sr=sample_rate)
            except Exception as e:
                print(f"⚠ skip (audio error: {e})")
                continue

            try:
                row          = tracks_df.loc[tid]
                title        = row[('track','title')]
                father_genre = row[('track','genre_top')]
                sub_genre    = row[('track','genres')]
                ex_bytes     = serialize_example(spec, tid, title, father_genre, sub_genre)
                writer.write(ex_bytes)
                written += 1
                print("✓")
            except Exception as e:
                print(f"⚠ skip (meta error: {e})")
                continue

        writer.close()
    print(f"> Done! Wrote {written}/{len(valid)} TFRecords to {output_path}")

if __name__ == '__main__':
    ROOT       = Path(__file__).resolve().parent.parent
    AUDIO_ZIP  = ROOT / 'data' / 'raw' / 'fma_medium.zip'
    OUTPUT_TF  = ROOT / 'data' / 'processed' / 'dataset.tfrecord'

    create_tfrecords(OUTPUT_TF, AUDIO_ZIP)