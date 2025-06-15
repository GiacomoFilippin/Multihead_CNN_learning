from pathlib import Path
import io, zipfile, sys
import tensorflow as tf
import numpy as np
import librosa

# inserisce scripts/ nel PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
from processing_functions import compute_mel_spectrogram, extract_track_id, load_features

def _bytes_feature(v: bytes)  -> tf.train.Feature: return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _int64_feature(v: int)   -> tf.train.Feature: return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

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
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()

def create_sharded_tfrecords(audio_zip_path: Path,
                             output_dir: Path,
                             shard_size: int = 500,
                             sample_rate: int = 44100):
    """
    Legge fma_medium.zip, calcola mel-spectrogram per ciascuna traccia
    e scrive in output_dir/shard_XXX.tfrecord, con XXX che avanza a ogni shard_size.
    """
    # 1) metadata
    tracks_df, _, _, _ = load_features()
    print(f"> Metadata loaded: {len(tracks_df)} tracks")

    # 2) prepara output
    output_dir.mkdir(exist_ok=True, parents=True)

    # 3) apri zip audio
    with zipfile.ZipFile(audio_zip_path, 'r') as zf:
        # filtra solo i file con metadati
        valid = []
        for name in zf.namelist():
            base = Path(name).name
            try:
                tid = int(extract_track_id(base))
            except:
                continue
            if tid in tracks_df.index:
                valid.append(name)
        print(f"> Found {len(valid)} audio files to shard")

        shard_idx = 0
        rec_in_shard = 0
        writer = None

        for i, arcname in enumerate(valid, 1):
            if rec_in_shard == 0:
                if writer: writer.close()
                shard_path = output_dir / f"dataset_{shard_idx:03d}.tfrecord"
                print(f"> Opening shard {shard_idx:03d}: {shard_path.name}")
                writer = tf.io.TFRecordWriter(str(shard_path))
                shard_idx += 1

            base = Path(arcname).name
            tid  = int(extract_track_id(base))
            print(f"[{i}/{len(valid)}] {base}", end=' → ')

            # load + spectrogram
            try:
                data = zf.read(arcname)
                y, _ = librosa.load(io.BytesIO(data), sr=sample_rate)
                spec = compute_mel_spectrogram(y, sr=sample_rate)
            except Exception as e:
                print(f"⚠ audio skip ({e})")
                continue

            # metadata + serialize
            try:
                row = tracks_df.loc[tid]
                ex = serialize_example(
                    spec, tid,
                    row[('track','title')],
                    row[('track','genre_top')],
                    row[('track','genres')]
                )
                writer.write(ex)
                rec_in_shard += 1
                print("✓")
            except Exception as e:
                print(f"⚠ meta skip ({e})")
                continue

            # se ho riempito lo shard, reset counter
            if rec_in_shard >= shard_size:
                rec_in_shard = 0

        if writer: writer.close()
    print(f"> Done! Created {shard_idx} shards in {output_dir}")

if __name__ == '__main__':
    ROOT       = Path(__file__).resolve().parent.parent
    AUDIO_ZIP  = ROOT / 'data' / 'raw' / 'fma_medium.zip'
    SHARDS_DIR = ROOT / 'data' / 'processed' / 'shards'
    create_sharded_tfrecords(AUDIO_ZIP, SHARDS_DIR, shard_size=64)