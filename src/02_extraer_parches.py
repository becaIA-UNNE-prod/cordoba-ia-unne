import os
import numpy as np
import rasterio
from rasterio.windows import Window

def extraer_parches_multitemporales(tile_id, ruta_base_s2, ruta_etiqueta, dir_salida, size=256):
    os.makedirs(dir_salida, exist_ok=True)

    ruta_tile = os.path.join(ruta_base_s2, tile_id)
    fechas = sorted([d for d in os.listdir(ruta_tile) if os.path.isdir(os.path.join(ruta_tile, d))])

    # Límite estricto de 3 fechas para prueba de concepto
    fechas = fechas[:3]
    bandas = ["B02", "B03", "B04", "B08"]

    print(f"Procesando Tile {tile_id} | Fechas a usar: {fechas}")
    print("Iniciando extracción por ventanas...")

    with rasterio.open(ruta_etiqueta) as src_label:
        alto, ancho = src_label.height, src_label.width
        contador_parches = 0

        for y in range(0, alto - size, size):
            for x in range(0, ancho - size, size):
                ventana = Window(x, y, size, size)
                parche_y = src_label.read(1, window=ventana)

                # Ignorar parches sin datos (0)
                if np.all(parche_y == 0):
                    continue

                parche_x_temporal = []
                for fecha in fechas:
                    for banda in bandas:
                        nombre_archivo = f"{tile_id}_{fecha}_{banda}.jp2"
                        ruta_banda = os.path.join(ruta_tile, fecha, nombre_archivo)

                        if os.path.exists(ruta_banda):
                            with rasterio.open(ruta_banda) as src_banda:
                                parche_banda = src_banda.read(1, window=ventana)
                                parche_x_temporal.append(parche_banda)
                        else:
                            parche_x_temporal.append(np.zeros((size, size), dtype=np.uint16))

                # Tensor X forma: (Fechas*Bandas, H, W)
                parche_x = np.stack(parche_x_temporal, axis=0)

                ruta_guardado_x = os.path.join(dir_salida, f"X_{tile_id}_{contador_parches}.npy")
                ruta_guardado_y = os.path.join(dir_salida, f"Y_{tile_id}_{contador_parches}.npy")

                np.save(ruta_guardado_x, parche_x.astype(np.float32))
                np.save(ruta_guardado_y, parche_y.astype(np.int64))

                contador_parches += 1
                if contador_parches % 20 == 0:
                    print(f"Extraídos {contador_parches} parches útiles...")

    print(f"Extracción finalizada. Total de parches generados: {contador_parches}")

if __name__ == "__main__":
    TILE_PRUEBA = "T20JLL"
    BASE_S2 = "/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba"
    RUTA_MASCARA = "./mascaras_procesadas/etiqueta_T20JLL_10m_test.tif"
    DIR_DATASET = "./dataset/train"

    if not os.path.exists(RUTA_MASCARA):
        raise FileNotFoundError(f"No se encontró: {RUTA_MASCARA}")

    extraer_parches_multitemporales(TILE_PRUEBA, BASE_S2, RUTA_MASCARA, DIR_DATASET)
