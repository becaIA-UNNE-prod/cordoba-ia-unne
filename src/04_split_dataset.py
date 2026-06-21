import os
import json
import random

def crear_split(dir_parches, ruta_salida, train=0.7, val=0.15, seed=42):
    archivos = sorted([f for f in os.listdir(dir_parches) if f.startswith('X_')])

    if not archivos:
        print(f"Error: No se encontraron parches en {dir_parches}")
        return

    random.seed(seed)
    random.shuffle(archivos)

    total = len(archivos)
    n_train = int(train * total)
    n_val = int(val * total)

    index = {
        "dir_parches": dir_parches,
        "seed": seed,
        "train": archivos[:n_train],
        "val":   archivos[n_train:n_train + n_val],
        "test":  archivos[n_train + n_val:],
    }

    with open(ruta_salida, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Split guardado en {ruta_salida}")
    print(f"Total: {total} | Train: {len(index['train'])} | Val: {len(index['val'])} | Test: {len(index['test'])}")

if __name__ == "__main__":
    DIR_PARCHES = "./dataset/train"
    RUTA_SPLIT  = "./dataset/split_index.json"

    crear_split(DIR_PARCHES, RUTA_SPLIT)
