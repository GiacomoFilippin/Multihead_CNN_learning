import random
from pathlib import Path
import tensorflow as tf

# parsing del TFRecord
from read_tfrecord import _parse_example
# dizionari id↔nome già generati
from configs.genres_dicts import father_name2id, subgenre_id2name

def _parse_and_preprocess(proto):
    parsed = _parse_example(proto)
    # 1) X = spectrogram with channel dimension
    spec = tf.expand_dims(parsed['spectrogram'], -1)  

    # 2) father genre → id via lookup table
    keys_f = tf.constant(list(father_name2id.keys()))
    vals_f = tf.constant(list(father_name2id.values()), dtype=tf.int64)
    table_f = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_f, vals_f),
        default_value=-1
    )
    father_id = table_f.lookup(parsed['father_genre'])

    # 3) sub_genre string "[1, 2]" → clean "1,2" → split → first → id
    sub_raw   = parsed['sub_genre']
    sub_clean = tf.strings.regex_replace(sub_raw, r'[\[\]\s]', '')  # remove brackets/spaces
    first     = tf.strings.split(sub_clean, ',')[0]
    sub_id    = tf.strings.to_number(first, out_type=tf.int64)

    # 4) return (X, y) where y is dict or tuple
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
    Restituisce (train_ds, val_ds, test_ds).  
    Ogni ds è un tf.data.Dataset di (spectrogram, targets) già batchato.
    """
    shards = sorted(Path(shards_dir).glob("*.tfrecord"))
    if not shards:
        raise FileNotFoundError(f"No shards in {shards_dir}")

    # split file list
    random.seed(seed)
    random.shuffle(shards)
    n = len(shards)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_files = shards[:n_train]
    val_files   = shards[n_train:n_train+n_val]
    test_files  = shards[n_train+n_val:]

    def _build_dataset(file_list):
        ds = tf.data.Dataset.from_tensor_slices([str(f) for f in file_list])
        ds = ds.interleave(
            lambda f: tf.data.TFRecordDataset(f),
            cycle_length=len(file_list),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.map(_parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(shuffle_buffer, seed=seed)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _build_dataset(train_files)
    val_ds   = _build_dataset(val_files)
    test_ds  = _build_dataset(test_files)

    return train_ds, val_ds, test_ds

""" example usage:
from scripts.dataset_loader import get_datasets

train_ds, val_ds, test_ds = get_datasets(
    shards_dir="data/processed/shards",
    batch_size=32
)

# durante il training:
x_batch, y_batch = next(iter(train_ds))
"""
if __name__ == "__main__":
    ROOT     = Path(__file__).resolve().parent.parent
    SHARDS   = ROOT / "data" / "processed" / "shards"
    train_ds, val_ds, test_ds = get_datasets(
        str(SHARDS), train_frac=0.8, val_frac=0.1, test_frac=0.1,
        batch_size=16, shuffle_buffer=1000
    )

    # test: prendi un batch
    x, y = next(iter(train_ds))
    print("Spectrogram batch shape:", x.shape)
    print("Father IDs:", y['father_genre'].numpy()[:5])
    print("Sub IDs:",    y['sub_genre'].numpy()[:5])