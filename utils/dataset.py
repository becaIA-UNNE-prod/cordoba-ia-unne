import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SentinelDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Listamos todos los archivos .npy de la carpeta images
        self.images_list = sorted(os.listdir(self.images_dir))
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        # 1. Nombres de archivo y rutas
        img_name = self.images_list[idx]
        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / img_name # Tienen el mismo nombre exacto
        
        # 2. Cargar los .npy desde el disco
        image = np.load(img_path)
        mask = np.load(mask_path)
        
        # 3. Normalización de Sentinel-2 (0 - 10000 -> 0.0 - 1.0)
        # Recortamos en 10000 por si hay valores anómalos o nubes muy brillantes
        image = np.clip(image, 0, 10000) 
        image = image.astype(np.float32) / 10000.0
        
        # 4. Data Augmentation (si está definido)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Si no hay transformaciones, solo convertimos a Tensores y reordenamos dimensiones
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.long)
            
        return image, mask

# --- CONFIGURACIÓN DE LOS DATALOADERS ---

def get_dataloaders(base_dir, batch_size=16):
    base_dir = Path(base_dir)
    
    # Definimos transformaciones para entrenamiento (Augmentation)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=False) # Convierte a tensor y ajusta dimensiones automáticamente
    ])

    # Para Validación y Test NO hacemos Augmentation, solo convertimos a tensor
    val_transform = A.Compose([
        ToTensorV2(transpose_mask=False)
    ])
    
    # Instanciamos los Datasets
    train_dataset = SentinelDataset(base_dir / "train" / "images", base_dir / "train" / "masks", transform=train_transform)
    val_dataset = SentinelDataset(base_dir / "val" / "images", base_dir / "val" / "masks", transform=val_transform)
    test_dataset = SentinelDataset(base_dir / "test" / "images", base_dir / "test" / "masks", transform=val_transform)
    
    # Creamos los DataLoaders
    # num_workers=4 permite cargar datos en paralelo usando el CPU mientras la GPU entrena
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# --- CÓMO USARLO ---
# BASE_DIR = "/mnt/yacy_1/prod/ferreyra/unet_dataset/"
# train_loader, val_loader, test_loader = get_dataloaders(BASE_DIR, batch_size=16)
