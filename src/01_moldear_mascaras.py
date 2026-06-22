import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def generar_etiquetas_como_npz(ruta_etiqueta_30m, ruta_referencia_s2, ruta_salida_npz):
    """
    Genera directamente la máscara en formato NPZ sin guardar TIF intermedio
    """
    print(f"Generando etiquetas en: {ruta_salida_npz}")

    # Leer metadatos de referencia (Sentinel-2)
    with rasterio.open(ruta_referencia_s2) as ref:
        crs_s2 = ref.crs
        transform_s2 = ref.transform
        width_s2 = ref.width
        height_s2 = ref.height

    # Reproject y leer directamente en memoria
    with rasterio.open(ruta_etiqueta_30m) as src:
        # Crear array destino
        etiquetas = np.zeros((height_s2, width_s2), dtype=np.int64)
        
        # Reproject directamente al array
        reproject(
            source=rasterio.band(src, 1),
            destination=etiquetas,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_s2,
            dst_crs=crs_s2,
            resampling=Resampling.nearest
        )
    
    # Guardar en formato comprimido
    np.savez_compressed(ruta_salida_npz, 
                       etiquetas=etiquetas,
                       crs=str(crs_s2),
                       transform=transform_s2,
                       width=width_s2,
                       height=height_s2)
    
    print(f"Etiquetas generadas. Shape: {etiquetas.shape}")
    print(f"Valores únicos: {np.unique(etiquetas)}")
    return ruta_salida_npz

def cargar_etiquetas_npz(ruta_npz):
    """
    Carga las etiquetas desde el archivo .npz
    """
    data = np.load(ruta_npz, allow_pickle=True)
    etiquetas = data['etiquetas']
    crs = str(data['crs'])
    transform = data['transform']
    width = int(data['width'])
    height = int(data['height'])
    
    print(f"Etiquetas cargadas. Shape: {etiquetas.shape}")
    print(f"Valores únicos: {np.unique(etiquetas)}")
    
    return etiquetas, crs, transform, width, height

if __name__ == "__main__":
    TIF_ORIGINAL = "./Nivel3_28_dic_2018_30m_completo.tif"
    DIR_SALIDA = "./mascaras_procesadas"
    os.makedirs(DIR_SALIDA, exist_ok=True)

    tile_prueba = "T20JLL"
    fecha_prueba = "20190108"
    banda_referencia = f"{tile_prueba}_{fecha_prueba}_B04.jp2"

    ruta_referencia = os.path.join(
        "/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba",
        tile_prueba,
        fecha_prueba,
        banda_referencia
    )

    ruta_salida_npz = os.path.join(DIR_SALIDA, f"etiqueta_{tile_prueba}.npz")

    if not os.path.exists(ruta_referencia):
        raise FileNotFoundError(f"No se encontró: {ruta_referencia}")

    # Generar directamente NPZ
    generar_etiquetas_como_npz(TIF_ORIGINAL, ruta_referencia, ruta_salida_npz)
    
    # Verificar carga
    etiquetas, crs, transform, width, height = cargar_etiquetas_npz(ruta_salida_npz)
    print(f"Verificación: {etiquetas.shape}, {width}x{height}")
