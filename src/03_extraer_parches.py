import os
import numpy as np
import rasterio
from rasterio.windows import Window

def extraer_parches_desde_composites(tile_id, ruta_mascara, dir_composites, dir_salida, size=256):
    os.makedirs(dir_salida, exist_ok=True)
    
    # Buscar todos los composites del tile y ordenarlos cronológicamente
    archivos_mensuales = sorted([f for f in os.listdir(dir_composites) if f.startswith(tile_id) and f.endswith('.tif')])
    
    if not archivos_mensuales:
        print(f"Error: No se encontraron composites para {tile_id} en {dir_composites}")
        return
        
    print(f"Iniciando extracción con {len(archivos_mensuales)} meses cronológicos.")

    with rasterio.open(ruta_mascara) as src_label:
        alto, ancho = src_label.height, src_label.width
        contador_parches = 0
        
        for y in range(0, alto - size, size):
            for x in range(0, ancho - size, size):
                ventana = Window(x, y, size, size)
                parche_y = src_label.read(1, window=ventana)
                
                # Ignorar parches sin cultivos (NoData)
                if np.all(parche_y == 0):
                    continue
                
                parche_x_temporal = []
                
                # Leer el mismo recuadro para cada mes
                for archivo_mes in archivos_mensuales:
                    ruta_mes = os.path.join(dir_composites, archivo_mes)
                    with rasterio.open(ruta_mes) as src_mes:
                        # Leemos todas las bandas (B02, B03, B04, B08) de este mes
                        parche_mes = src_mes.read(window=ventana) 
                        parche_x_temporal.append(parche_mes)
                
                # Apilar todo. Forma resultante: (Meses * Bandas, H, W)
                parche_x = np.concatenate(parche_x_temporal, axis=0)
                
                ruta_guardado_x = os.path.join(dir_salida, f"X_{tile_id}_{contador_parches}.npy")
                ruta_guardado_y = os.path.join(dir_salida, f"Y_{tile_id}_{contador_parches}.npy")
                
                np.save(ruta_guardado_x, parche_x.astype(np.float32))
                np.save(ruta_guardado_y, parche_y.astype(np.int64))
                
                contador_parches += 1
                if contador_parches % 50 == 0:
                    print(f"Extraídos {contador_parches} parches útiles...")

    print(f"Extracción finalizada. Total: {contador_parches} parches.")

if __name__ == "__main__":
    TILE_PRUEBA = "T20JLL"
    RUTA_MASCARA = "./mascaras_procesadas/etiqueta_T20JLL_10m_test.tif" 
    DIR_COMPOSITES = "./dataset/composites"
    DIR_DATASET = "./dataset/train" 
    
    extraer_parches_desde_composites(TILE_PRUEBA, RUTA_MASCARA, DIR_COMPOSITES, DIR_DATASET)
