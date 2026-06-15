import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CordobaDataset(Dataset):
    def __init__(self, dir_parches):
        self.dir_parches = dir_parches
        # Buscar solo los archivos X y ordenarlos para asegurar correspondencia
        self.archivos_x = sorted([f for f in os.listdir(dir_parches) if f.startswith('X_')])

    def __len__(self):
        return len(self.archivos_x)

    def __getitem__(self, idx):
        nombre_x = self.archivos_x[idx]
        nombre_y = nombre_x.replace('X_', 'Y_')

        ruta_x = os.path.join(self.dir_parches, nombre_x)
        ruta_y = os.path.join(self.dir_parches, nombre_y)

        # Cargar matrices
        array_x = np.load(ruta_x).astype(np.float32)
        array_y = np.load(ruta_y).astype(np.int64)

        # Normalización simple para Sentinel-2 (valores L2A suelen estar entre 0 y 10000)
        array_x = array_x / 10000.0

        # Convertir a tensores de PyTorch
        tensor_x = torch.from_numpy(array_x)
        tensor_y = torch.from_numpy(array_y)

        return tensor_x, tensor_y
