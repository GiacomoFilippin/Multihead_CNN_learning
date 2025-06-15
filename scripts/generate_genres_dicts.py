from pathlib import Path
import sys
import pandas as pd

# import load_features per leggere CSV raw
sys.path.insert(0, str(Path(__file__).parent))
from processing_functions import load_features

# 1) carico i DataFrame
tracks_df, genres_df, _, _ = load_features()

# 2) preparo i dizionari
subgenre_id2name   = genres_df["title"].to_dict()
subgenre_name2id   = {v: k for k, v in subgenre_id2name.items()}
unique_fathers     = tracks_df[("track", "genre_top")].unique().tolist()
father_name2id     = {name: idx for idx, name in enumerate(unique_fathers)}
father_id2name     = {idx: name for name, idx in father_name2id.items()}

# 3) scrivo il modulo configs/genres_dicts.py
cfg_dir = Path(__file__).parent.parent / "configs"
cfg_dir.mkdir(exist_ok=True)
out = cfg_dir / "genres_dicts.py"

with open(out, "w", encoding="utf-8") as f:
    f.write("# Auto‐generated: non modificare a mano\n\n")
    f.write(f"subgenre_id2name   = {repr(subgenre_id2name)}\n\n")
    f.write(f"subgenre_name2id   = {repr(subgenre_name2id)}\n\n")
    f.write(f"father_name2id     = {repr(father_name2id)}\n\n")
    f.write(f"father_id2name     = {repr(father_id2name)}\n")
print(f"Wrote genre dicts to {out}")