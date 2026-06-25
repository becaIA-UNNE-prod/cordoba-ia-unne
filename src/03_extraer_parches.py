import os
import numpy as np
import rasterio
from rasterio.windows import Window
import time

def extraer_parches_desde_composites(tile_id, ruta_mascara, dir_composites, dir_salida, size=256, step=None):
    os.makedirs(dir_salida, exist_ok=True)
    
    if step is None:
        step = size
    
    archivos_mensuales = sorted([f for f in os.listdir(dir_composites) 
                                 if f.startswith(tile_id) and f.endswith('.tif')])
    
    if not archivos_mensuales:
        print(f"Error: No se encontraron composites para {tile_id}")
        return None
    
    parches_x = []
    parches_y = []
    
    with rasterio.open(ruta_mascara) as src_label:
        alto, ancho = src_label.height, src_label.width
        contador_parches = 0
        # Calcular número total de parches (aproximado)
        n_parches_x = (ancho - size) // step + 1
        n_parches_y = (alto - size) // step + 1
        total_posibles = n_parches_x * n_parches_y
        
        print(f"Recorriendo {n_parches_x}×{n_parches_y} = {total_posibles:,} parches...")
        
        start_time = time.time()
        
        for y in range(0, alto - size, step):
            for x in range(0, ancho - size, step):
                ventana = Window(x, y, size, size)
                parche_y = src_label.read(1, window=ventana)
                
                if np.all(parche_y == 0):
                    continue
                
                parche_x_temporal = []
                for archivo_mes in archivos_mensuales:
                    ruta_mes = os.path.join(dir_composites, archivo_mes)
                    with rasterio.open(ruta_mes) as src_mes:
                        parche_mes = src_mes.read(window=ventana)
                        parche_x_temporal.append(parche_mes)
                
                parche_x = np.concatenate(parche_x_temporal, axis=0)
                parches_x.append(parche_x.astype(np.float32))
                parches_y.append(parche_y.astype(np.int64))
                contador_parches += 1
                if contador_parches % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f" Procesados {contador_parches} parches en {elapsed:.1f}s...")
                          
    if contador_parches == 0:
        print(f"Advertencia: No se encontraron parches con cultivos")
        return None
    
    X = np.stack(parches_x, axis=0)
    Y = np.stack(parches_y, axis=0)
    
    os.makedirs(dir_salida, exist_ok=True)
    ruta_salida = os.path.join(dir_salida, f"dataset_{tile_id}.npz")
    np.savez_compressed(ruta_salida, X=X, Y=Y, meses=archivos_mensuales, tile_id=tile_id)
    
    print(f"Guardados {contador_parches} parches en {ruta_salida}")
    return ruta_salida
# crear_posiciones.py
import numpy as np
import os

def crear_posiciones(dir_salida, size=256, step=192, alto=10980, ancho=10980):
    """
    Crea el archivo posiciones_parches.npy manualmente
    Asume que los parches se extrajeron en orden de escaneo
    """
    posiciones = []
    for y in range(0, alto - size, step):
        for x in range(0, ancho - size, step):
            posiciones.append((x, y))
    
    os.makedirs(dir_salida, exist_ok=True)
    np.save(os.path.join(dir_salida, "posiciones_parches.npy"), np.array(posiciones))
    print(f"Posiciones guardadas en {dir_salida}/posiciones_parches.npy")
    print(f"Total: {len(posiciones)} posiciones")

    
def cargar_dataset(ruta_npz):
    data = np.load(ruta_npz, allow_pickle=True)
    X = data['X']
    Y = data['Y']
    meses = data['meses']
    tile_id = str(data['tile_id'])
    return X, Y, meses, tile_id

if __name__ == "__main__":
    
    BASE_DIR = "/mnt/yacy_1/prod/ferreyra/dataset"
    TILE_PRUEBA = "T20JLL"
    RUTA_MASCARA = f"{BASE_DIR}/etiquetas/etiqueta_20JLL_10m_test.tif"
    DIR_COMPOSITES = f"{BASE_DIR}/composites"
    DIR_DATASET = f"{BASE_DIR}/train"
    crear_posiciones(DIR_DATASET, size=256, step=256)
    archivo_salida = extraer_parches_desde_composites(
        TILE_PRUEBA, 
        RUTA_MASCARA, 
        DIR_COMPOSITES, 
        DIR_DATASET,
        size=256,
        step=256
    )
    
    if archivo_salida:
        X, Y, meses, tile = cargar_dataset(archivo_salida)
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"Meses: {len(meses)}")
        
