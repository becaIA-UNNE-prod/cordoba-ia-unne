import os
from pathlib import Path
import rasterio
from rasterio.warp import reproject, Resampling

# --- MODO DEBUG ---
DEBUG_MODE = False
# ------------------

BASE_DIR = Path("/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba/")
RUTA_MASCARA_GRANDE = Path("cobertura_y_uso_2021.tif")
MASK_TILES_DIR = Path("/mnt/yacy_1/prod/ferreyra/mascaras_por_tile/")

MASK_TILES_DIR.mkdir(parents=True, exist_ok=True)

todos_los_tiles = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]

# Lógica Debug: Limitar a 1 solo Tile
if DEBUG_MODE:
    print(f"⚠️ MODO DEBUG ACTIVO: Solo se procesará 1 Tile de los {len(todos_los_tiles)} disponibles.")
    todos_los_tiles = todos_los_tiles[:1]

print(f"Abriendo máscara original gigante...")
with rasterio.open(RUTA_MASCARA_GRANDE) as src_big_mask:
    
    for tile in todos_los_tiles:
        tile_path = BASE_DIR / tile
        fechas = [d for d in tile_path.iterdir() if d.is_dir()]
        if not fechas: 
            continue
        
        ref_band_path = None
        for f_dir in fechas:
            found = list(f_dir.glob(f"{tile}_*_B02.jp2"))
            if found:
                ref_band_path = found[0]
                break
        
        if not ref_band_path:
            print(f"No se encontró banda B02 para el Tile {tile}. Saltando...")
            continue
            
        with rasterio.open(ref_band_path) as src_ref:
            kwargs = src_ref.meta.copy()
            
            kwargs.update({
                'driver': 'GTiff',
                'count': 1,
                'dtype': src_big_mask.dtypes[0], 
                'nodata': src_big_mask.nodata if src_big_mask.nodata is not None else 0
            })
            
            output_tile_mask = MASK_TILES_DIR / f"mascara_{tile}.tif"
            print(f"-> Creando máscara geométrica para: {tile}...")
            
            with rasterio.open(output_tile_mask, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src_big_mask, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src_big_mask.transform,
                    src_crs=src_big_mask.crs,
                    dst_transform=src_ref.transform,
                    dst_crs=src_ref.crs,
                    resampling=Resampling.nearest 
                )

print("\n¡PASO 1 FINALIZADO!")
