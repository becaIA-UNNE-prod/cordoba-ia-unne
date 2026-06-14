import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import rasterio
from rasterio.windows import Window

# --- MODO DEBUG ---
DEBUG_MODE = False
MAX_PATCHES_DEBUG = 10  # Cuántos parches extraer por imagen en modo debug
# ------------------

BASE_DIR = Path("/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba/")
MASK_TILES_DIR = Path("/mnt/yacy_1/prod/ferreyra/mascaras_por_tile/")
OUTPUT_DIR = Path("/mnt/yacy_1/prod/ferreyra/unet_dataset/")
PATCH_SIZE = 256

todos_los_tiles = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]

train_tiles, temp_tiles = train_test_split(todos_los_tiles, test_size=0.30, random_state=42)
val_tiles, test_tiles = train_test_split(temp_tiles, test_size=0.50, random_state=42)

# Lógica Debug: Limitar a 1 Tile por partición
if DEBUG_MODE:
    print(f"⚠️ MODO DEBUG ACTIVO: Forzando el uso del Tile T20JKP.")
    train_tiles = ["T20JKP"]
    val_tiles = []
    test_tiles = []

splits = {'train': train_tiles, 'val': val_tiles, 'test': test_tiles}

for split_name in splits.keys():
    (OUTPUT_DIR / split_name / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split_name / "masks").mkdir(parents=True, exist_ok=True)

for split_name, tiles in splits.items():
    print(f"\nProcesando conjunto: {split_name.upper()}")
    for tile in tiles:
        tile_path = BASE_DIR / tile
        tile_mask_path = MASK_TILES_DIR / f"mascara_{tile}.tif"

        if not tile_mask_path.exists():
            print(f"Máscara para {tile} no existe, saltando...")
            continue

        for date_dir in tile_path.iterdir():
            if not date_dir.is_dir(): continue
            fecha = date_dir.name

            if not (fecha.startswith("2020") or fecha.startswith("2021")):
                continue

            b2_p = date_dir / f"{tile}_{fecha}_B02.jp2"
            b3_p = date_dir / f"{tile}_{fecha}_B03.jp2"
            b4_p = date_dir / f"{tile}_{fecha}_B04.jp2"
            b8_p = date_dir / f"{tile}_{fecha}_B08.jp2"

            if not all(p.exists() for p in [b2_p, b3_p, b4_p, b8_p]):
                continue

            try:
                with rasterio.open(b2_p) as src_b2, rasterio.open(b3_p) as src_b3, \
                     rasterio.open(b4_p) as src_b4, rasterio.open(b8_p) as src_b8, \
                     rasterio.open(tile_mask_path) as src_m:
                    
                    height, width = src_b2.shape
                    
                    for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
                        for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
                            window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                            
                            b2 = src_b2.read(1, window=window)
                            b3 = src_b3.read(1, window=window)
                            b4 = src_b4.read(1, window=window)
                            b8 = src_b8.read(1, window=window)
                            mask_patch = src_m.read(1, window=window)
                            
                            img_patch = np.dstack((b2, b3, b4, b8))
                            
                            if np.max(img_patch) == 0:
                                continue
                                
                            filename = f"{tile}_{fecha}_y{y}_x{x}.npy"
                            np.save(OUTPUT_DIR / split_name / "images" / filename, img_patch)
                            np.save(OUTPUT_DIR / split_name / "masks" / filename, mask_patch)
                            
            except Exception as e:
                print(f"⚠️ Archivo corrupto detectado en {tile} - {fecha}. Saltando... (Detalle: {e})")
                continue # Pasa a la siguiente iteración de fecha automáticamente

            print(f"  -> Extrayendo de {tile} - {fecha}")
            patches_extraidos = 0

            with rasterio.open(b2_p) as src_b2, rasterio.open(b3_p) as src_b3, \
                 rasterio.open(b4_p) as src_b4, rasterio.open(b8_p) as src_b8, \
                 rasterio.open(tile_mask_path) as src_m:

                height, width = src_b2.shape

                for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
                    if DEBUG_MODE and patches_extraidos >= MAX_PATCHES_DEBUG: break

                    for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
                        if DEBUG_MODE and patches_extraidos >= MAX_PATCHES_DEBUG: break

                        window = Window(x, y, PATCH_SIZE, PATCH_SIZE)

                        b2 = src_b2.read(1, window=window)
                        b3 = src_b3.read(1, window=window)
                        b4 = src_b4.read(1, window=window)
                        b8 = src_b8.read(1, window=window)
                        mask_patch = src_m.read(1, window=window)

                        img_patch = np.dstack((b2, b3, b4, b8))

                        if np.max(img_patch) == 0:
                            continue

                        filename = f"{tile}_{fecha}_y{y}_x{x}.npy"
                        np.save(OUTPUT_DIR / split_name / "images" / filename, img_patch)
                        np.save(OUTPUT_DIR / split_name / "masks" / filename, mask_patch)

                        patches_extraidos += 1

            if DEBUG_MODE:
                print(f"     Extraídos {patches_extraidos} parches en modo debug.")
                break # Rompe el loop de fechas para saltar rápido al siguiente split

print("\n¡PASO 2 COMPLETADO EN MODO DEBUG!")
