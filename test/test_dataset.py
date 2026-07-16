"""
Test de humo para CordobaDataset (utils/dataset.py), sin depender de datos
reales ni del servidor: genera tiles sinteticos con la misma pinta que
produce src/03_extraer_parches.py (X_<tile>.npy, Y_<tile>.npy, meta_<tile>.npz,
posiciones_<tile>.npy) y verifica que el dataset multi-tile:

  - no requiere que ningun tile completo este en RAM (usa mmap)
  - resuelve correctamente los indices globales a (tile, indice local)
  - aplica remapeo de etiquetas y normalizacion
  - detecta con un error claro tiles con distinta cantidad de canales

Correr con: cd test && python test_dataset.py
(o con pytest, si esta instalado: pytest test/test_dataset.py)
"""
import os
import sys
import shutil
import tempfile

import numpy as np

sys.path.append(os.path.abspath(".."))
from utils.dataset import CordobaDataset


def crear_tile_sintetico(dir_dataset, tile_id, n_parches, canales, size=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10000, size=(n_parches, canales, size, size)).astype(np.float32)
    Y = rng.integers(0, 28, size=(n_parches, size, size)).astype(np.int64)
    meses = [f"{tile_id}_2018{m:02d}_median.tif" for m in range(1, canales // 4 + 1)]
    posiciones = np.array([(i * size, 0) for i in range(n_parches)], dtype=np.int64)

    np.save(os.path.join(dir_dataset, f"X_{tile_id}.npy"), X)
    np.save(os.path.join(dir_dataset, f"Y_{tile_id}.npy"), Y)
    np.savez(os.path.join(dir_dataset, f"meta_{tile_id}.npz"),
             meses=meses, tile_id=tile_id, n_samples=n_parches)
    np.save(os.path.join(dir_dataset, f"posiciones_{tile_id}.npy"), posiciones)
    return X, Y


def test_multi_tile_indexing_y_mmap():
    dir_tmp = tempfile.mkdtemp(prefix="cordoba_test_")
    try:
        X_a, Y_a = crear_tile_sintetico(dir_tmp, "T_A", n_parches=5, canales=8, seed=1)
        X_b, Y_b = crear_tile_sintetico(dir_tmp, "T_B", n_parches=3, canales=8, seed=2)

        ds = CordobaDataset(dir_tmp, ["T_A", "T_B"], normalizar=True, label_map=None)

        assert len(ds) == 8, f"esperaba 8 parches totales, dio {len(ds)}"
        assert ds.in_channels == 8

        # Los arrays subyacentes deben seguir siendo memmap (no cargados a RAM)
        assert isinstance(ds._X[0], np.memmap), "X del tile A deberia seguir siendo un memmap"
        assert isinstance(ds._X[1], np.memmap), "X del tile B deberia seguir siendo un memmap"

        # Indice 0..4 -> tile A, 5..7 -> tile B
        for i in range(5):
            tile_id, local_idx = ds.tile_de_indice(i)
            assert tile_id == "T_A" and local_idx == i, (i, tile_id, local_idx)
        for i in range(5, 8):
            tile_id, local_idx = ds.tile_de_indice(i)
            assert tile_id == "T_B" and local_idx == i - 5, (i, tile_id, local_idx)

        assert ds.indices_de_tile("T_A") == list(range(0, 5))
        assert ds.indices_de_tile("T_B") == list(range(5, 8))

        # Contenido correcto y normalizado (idx 6 -> tile B, local 1)
        x, y = ds[6]
        esperado = X_b[1] / 10000.0
        assert np.allclose(x.numpy(), esperado), "el parche recuperado no coincide con el original (normalizado)"
        assert np.array_equal(y.numpy(), Y_b[1])

        print("OK: indexado multi-tile, mmap y normalizacion")
    finally:
        shutil.rmtree(dir_tmp)


def test_label_remap():
    dir_tmp = tempfile.mkdtemp(prefix="cordoba_test_")
    try:
        crear_tile_sintetico(dir_tmp, "T_A", n_parches=2, canales=4, seed=3)
        label_map = np.zeros(28, dtype=np.int64)
        label_map[15] = 3  # como en Cnf.label_remap: trigo -> clase 3

        ds = CordobaDataset(dir_tmp, ["T_A"], normalizar=False, label_map=label_map)
        _, y = ds[0]
        assert set(np.unique(y.numpy())).issubset({0, 3}), "el remapeo de etiquetas no se aplico"
        print("OK: remapeo de etiquetas")
    finally:
        shutil.rmtree(dir_tmp)


def test_error_claro_si_canales_no_coinciden():
    dir_tmp = tempfile.mkdtemp(prefix="cordoba_test_")
    try:
        crear_tile_sintetico(dir_tmp, "T_A", n_parches=2, canales=8, seed=4)
        crear_tile_sintetico(dir_tmp, "T_B", n_parches=2, canales=12, seed=5)  # otra cantidad de meses

        try:
            CordobaDataset(dir_tmp, ["T_A", "T_B"])
            assert False, "deberia haber fallado por canales distintos entre tiles"
        except ValueError as e:
            assert "canales" in str(e)
            print("OK: error claro por cantidad de canales distinta entre tiles")
    finally:
        shutil.rmtree(dir_tmp)


def test_error_claro_si_falta_archivo():
    dir_tmp = tempfile.mkdtemp(prefix="cordoba_test_")
    try:
        try:
            CordobaDataset(dir_tmp, ["T_NO_EXISTE"])
            assert False, "deberia haber fallado por archivo faltante"
        except FileNotFoundError:
            print("OK: error claro por archivo faltante")
    finally:
        shutil.rmtree(dir_tmp)


if __name__ == "__main__":
    test_multi_tile_indexing_y_mmap()
    test_label_remap()
    test_error_claro_si_canales_no_coinciden()
    test_error_claro_si_falta_archivo()
    print("\nTodos los tests pasaron.")
