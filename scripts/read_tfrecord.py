from pathlib import Path
import random
import tensorflow as tf
import sys
import os
ROOT = Path(__file__).resolve().parent.parent  
sys.path.insert(0, str(ROOT))
# per il lookup dei generi
from configs.genres_dicts import father_name2id

# 1) Definizione della struttura del TFRecord
FEATURE_DESCRIPTION = {
    'spectrogram':   tf.io.FixedLenFeature([], tf.string),
    'height':        tf.io.FixedLenFeature([], tf.int64),
    'width':         tf.io.FixedLenFeature([], tf.int64),
    'track_id':      tf.io.FixedLenFeature([], tf.int64),
    'title':         tf.io.FixedLenFeature([], tf.string),
    'father_genre':  tf.io.FixedLenFeature([], tf.string),
    'sub_genre':     tf.io.FixedLenFeature([], tf.string),
}

def _parse_tfrecord(example_proto):
    """
    1) parse raw TFRecord
    2) ricostruisce lo spettrogramma (h, w, 1)
    3) converte father_genre→id numerico
    4) estrae il primo sub_genre→id
    """
    parsed = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

    # decode spectrogram
    spec_raw = tf.io.decode_raw(parsed['spectrogram'], tf.float32)
    h = tf.cast(parsed['height'], tf.int32)
    w = tf.cast(parsed['width'],  tf.int32)
    spec = tf.reshape(spec_raw, (h, w))
    spec = tf.expand_dims(spec, axis=-1)   # canale=1

    # father_genre string → id con StaticHashTable
    keys_f = tf.constant(list(father_name2id.keys()))
    vals_f = tf.constant(list(father_name2id.values()), dtype=tf.int64)
    table_f = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_f, vals_f),
        default_value=-1
    )
    father_id = table_f.lookup(parsed['father_genre'])

    # sub_genre string "[1, 2]" → remove brackets → split → take [0] → int
    sg = tf.strings.regex_replace(parsed['sub_genre'], r'[\[\]\s]', '')
    first = tf.strings.split(sg, ',')[0]
    sub_id = tf.strings.to_number(first, out_type=tf.int64)

    # X = spectrogram, y = dict con due label
    y = {
        'father_genre': father_id,
        'sub_genre':    sub_id
    }
    return spec, y

def get_datasets(shards_dir: str,
                 train_frac: float = 0.8,
                 val_frac: float   = 0.1,
                 test_frac: float  = 0.1,
                 batch_size: int   = 32,
                 shuffle_buffer: int = 1000,
                 seed: int = 42):
    """
    Ritorna train_ds, val_ds, test_ds già batchati.
    Ogni elemento è (spectrogram, {father_genre, sub_genre}).
    """
    shards = sorted(Path(shards_dir).glob("*.tfrecord"))
    if not shards:
        raise FileNotFoundError(f"No shards in {shards_dir}")

    # shuffle shards list
    random.seed(seed)
    random.shuffle(shards)
    n = len(shards)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_files = shards[:n_train]
    val_files   = shards[n_train:n_train + n_val]
    test_files  = shards[n_train + n_val:]

    def _build_ds(files):
        ds = tf.data.Dataset.from_tensor_slices([str(f) for f in files])
        ds = ds.interleave(
            lambda f: tf.data.TFRecordDataset(f),
            cycle_length=len(files),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(shuffle_buffer, seed=seed)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _build_ds(train_files)
    val_ds   = _build_ds(val_files)
    test_ds  = _build_ds(test_files)

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    ROOT    = Path(__file__).resolve().parent.parent
    SHARDS  = ROOT / "data" / "processed" / "shards"

    train_ds, val_ds, test_ds = get_datasets(
        str(SHARDS),
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        batch_size=16
    )

    # sanity check
    x, y = next(iter(train_ds))
    print("X batch shape:", x.shape)
    print("Father IDs:", y['father_genre'].numpy()[:5])
    print("Sub IDs:",    y['sub_genre'].numpy()[:5])