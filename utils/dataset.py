import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CordobaDataset(Dataset):
    def __init__(self, ruta_npz, normalizar=True):
        """
        Dataset para clasificación de cultivos con Sentinel-2
        
        Args:
            ruta_npz: Ruta al archivo .npz con los datos
            normalizar: Si True, normaliza los valores dividiendo por 10000
        """
        self.ruta_npz = ruta_npz
        self.normalizar = normalizar
        
        print(f"Cargando dataset desde: {ruta_npz}")
        data = np.load(ruta_npz, allow_pickle=True)
        
        self.X = data['X'].astype(np.float32)
        self.Y = data['Y'].astype(np.int64)
        self.meses = data['meses']
        self.tile_id = str(data['tile_id'])
        
        self.n_samples = self.X.shape[0]
        print(f"Dataset cargado. Samples: {self.n_samples}")
        print(f"X shape: {self.X.shape}")
        print(f"Y shape: {self.Y.shape}")
        print(f"Valores únicos en Y: {np.unique(self.Y)}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        array_x = self.X[idx]
        array_y = self.Y[idx]
        
        # Normalización (opcional)
        if self.normalizar:
            array_x = array_x / 10000.0
        
        # Convertir a tensores de PyTorch
        tensor_x = torch.from_numpy(array_x)
        tensor_y = torch.from_numpy(array_y)
        
        return tensor_x, tensor_y

    def get_metadata(self):
        """Retorna metadatos del dataset"""
        return {
            'tile_id': self.tile_id,
            'n_samples': self.n_samples,
            'shape_x': self.X.shape,
            'shape_y': self.Y.shape,
            'meses': self.meses,
            'valores_y': np.unique(self.Y)
        }
