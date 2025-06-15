from pathlib import Path
import tensorflow as tf

def split_tfrecord(input_tfrecord: Path,
                   output_dir: Path,
                   chunk_size: int = 1000):
    """
    Divide input_tfrecord in file da `chunk_size` record ciascuno.
    I nuovi file saranno chiamati shard_000.tfrecord, shard_001.tfrecord, ...
    """
    output_dir.mkdir(exist_ok=True)
    ds = tf.data.TFRecordDataset(str(input_tfrecord))
    shard_idx = 0
    rec_in_shard = 0
    writer = None

    for record in ds:
        # apri un nuovo writer all'inizio e ogni volta che raggiungiamo chunk_size
        if rec_in_shard == 0:
            if writer:
                writer.close()
            shard_path = output_dir / f"shard_{shard_idx:03d}.tfrecord"
            print(f"> Creating {shard_path} …")
            writer = tf.io.TFRecordWriter(str(shard_path))
            shard_idx += 1

        writer.write(record.numpy())
        rec_in_shard += 1

        if rec_in_shard >= chunk_size:
            rec_in_shard = 0

    # chiudi l'ultimo writer
    if writer:
        writer.close()
    print(f"> Done! Produced {shard_idx} shards in {output_dir}")

if __name__ == "__main__":
    ROOT      = Path(__file__).resolve().parent.parent
    INPUT_TF  = ROOT / "data" / "processed" / "dataset.tfrecord"
    OUTPUT_SH = ROOT / "data" / "processed" / "shards"
    CHUNK     = 500   # numero di canzoni per shard

    split_tfrecord(INPUT_TF, OUTPUT_SH, chunk_size=CHUNK)