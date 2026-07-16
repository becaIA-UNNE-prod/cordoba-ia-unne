import os
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset


class CordobaDataset(Dataset):
    """
    Dataset multi-tile para clasificacion de cultivos con Sentinel-2.

    Cada tile vive en 3 archivos dentro de `dir_dataset`:
        X_<tile>.npy    parches de imagen   (N, canales, H, W) float32
        Y_<tile>.npy    parches de etiqueta (N, H, W)          int64
        meta_<tile>.npz metadata liviana (meses, tile_id, n_samples)

    Por que .npy y no .npz para X/Y: numpy permite abrir un .npy con
    mmap_mode='r', de forma que el archivo se queda en disco y el sistema
    operativo pagina a RAM unicamente los parches que se piden en cada
    __getitem__. Un .npz (comprimido o no) NO soporta esto: acceder a
    data['X'] siempre descomprime/copia el array COMPLETO a memoria
    (NpzFile.__getitem__ llama a format.read_array sobre el stream del
    zip, no hay ruta de mmap). Por eso, con varios tiles cargados a la
    vez, un dataset basado en .npz escala en RAM con la cantidad de
    tiles; este, basado en .npy memory-mapped, no: el consumo de RAM
    depende del tamano de los batches, no de cuantos tiles se agreguen
    a `tiles`.
    """

    def __init__(self, dir_dataset, tiles, normalizar=True, label_map=None):
        if isinstance(tiles, str):
            tiles = [tiles]
        tiles = list(tiles)
        if not tiles:
            raise ValueError("`tiles` no puede estar vacio")

        self.dir_dataset = dir_dataset
        self.tiles = tiles
        self.normalizar = normalizar
        self.label_map = label_map

        self._X = []
        self._Y = []
        self.tile_ids = []
        self.meses_por_tile = []
        tile_lengths = []
        in_channels_ref = None
        patch_shape_ref = None

        for tile in tiles:
            x_path = os.path.join(dir_dataset, f"X_{tile}.npy")
            y_path = os.path.join(dir_dataset, f"Y_{tile}.npy")
            meta_path = os.path.join(dir_dataset, f"meta_{tile}.npz")

            for p in (x_path, y_path, meta_path):
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"Falta '{p}' para el tile '{tile}'. "
                        f"Corre src/03_extraer_parches.py para ese tile antes de entrenar."
                    )

            # mmap_mode='r': no carga nada a RAM todavia, solo abre el archivo
            # y lee el header (shape/dtype).
            X = np.load(x_path, mmap_mode='r')
            Y = np.load(y_path, mmap_mode='r')
            meta = np.load(meta_path, allow_pickle=True)

            if X.shape[0] != Y.shape[0]:
                raise ValueError(
                    f"Tile '{tile}': X tiene {X.shape[0]} parches pero Y tiene {Y.shape[0]}"
                )

            n_channels = X.shape[1]
            patch_shape = X.shape[2:]
            if in_channels_ref is None:
                in_channels_ref = n_channels
                patch_shape_ref = patch_shape
            elif n_channels != in_channels_ref:
                raise ValueError(
                    f"Tile '{tile}' tiene {n_channels} canales (meses x bandas), pero los "
                    f"tiles anteriores en la lista tienen {in_channels_ref}. Todos los tiles "
                    f"deben cubrir los mismos meses para poder entrenar juntos con un unico "
                    f"modelo (los canales de entrada de la UNet son fijos)."
                )
            elif patch_shape != patch_shape_ref:
                raise ValueError(
                    f"Tile '{tile}' tiene parches de tamano {patch_shape}, pero los tiles "
                    f"anteriores tienen {patch_shape_ref}."
                )

            print(f"Tile '{tile}': {X.shape[0]} parches, {n_channels} canales, "
                  f"{patch_shape[0]}x{patch_shape[1]} px (mmap, no cargado a RAM)")

            self._X.append(X)
            self._Y.append(Y)
            self.tile_ids.append(str(meta['tile_id']))
            self.meses_por_tile.append(meta['meses'])
            tile_lengths.append(int(X.shape[0]))

        self.tile_lengths = tile_lengths
        self.cum_lengths = np.cumsum([0] + tile_lengths)
        self.n_samples = int(self.cum_lengths[-1])
        self.in_channels = in_channels_ref

        print(f"Dataset multi-tile listo. Tiles: {self.tile_ids} | "
              f"Total parches: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def _resolver_indice(self, idx):
        if idx < 0:
            idx += self.n_samples
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"Indice {idx} fuera de rango (0..{self.n_samples - 1})")
        tile_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        local_idx = idx - self.cum_lengths[tile_idx]
        return tile_idx, int(local_idx)

    def tile_de_indice(self, idx):
        """ tile_id e indice local (dentro del tile) correspondientes a un indice global. """
        tile_idx, local_idx = self._resolver_indice(idx)
        return self.tile_ids[tile_idx], local_idx

    def indices_de_tile(self, tile_id):
        """ Lista de indices globales que pertenecen a un tile dado. """
        tile_idx = self.tile_ids.index(tile_id)
        inicio = int(self.cum_lengths[tile_idx])
        fin = int(self.cum_lengths[tile_idx + 1])
        return list(range(inicio, fin))

    def __getitem__(self, idx):
        tile_idx, local_idx = self._resolver_indice(idx)

        # np.array(...) copia el parche fuera del mmap: recien aca el parche
        # pedido pasa a vivir en RAM normal (un solo parche, no todo el tile).
        array_x = np.array(self._X[tile_idx][local_idx], dtype=np.float32)
        array_y = np.array(self._Y[tile_idx][local_idx], dtype=np.int64)

        if self.label_map is not None:
            array_y = self.label_map[array_y]

        if self.normalizar:
            array_x = array_x / 10000.0

        return torch.from_numpy(array_x), torch.from_numpy(array_y)

    def get_metadata(self):
        return {
            'tiles': self.tile_ids,
            'n_samples': self.n_samples,
            'tile_lengths': self.tile_lengths,
            'in_channels': self.in_channels,
            'meses_por_tile': self.meses_por_tile,
        }
