import os
import rasterio
import numpy as np
from collections import defaultdict
import concurrent.futures

def procesar_un_mes(args):
    """
    Función aislada que procesa un solo mes.
    Ideal para ser ejecutada por un worker en paralelo.
    """
    mes, fechas_del_mes, tile_id, ruta_tile, dir_salida, bandas = args
    ruta_salida_archivo = os.path.join(dir_salida, f"{tile_id}_{mes}_median.tif")

    if os.path.exists(ruta_salida_archivo):
        return f"[{mes}] Ya existe. Omitiendo."

    print(f"[{mes}] Iniciando cálculo con {len(fechas_del_mes)} fechas...")

    # 1. Leer metadatos de la primera imagen disponible
    ruta_plantilla = os.path.join(ruta_tile, fechas_del_mes[0], f"{tile_id}_{fechas_del_mes[0]}_B02.jp2")
    with rasterio.open(ruta_plantilla) as src_ref:
        meta = src_ref.meta.copy()
        meta.update(count=len(bandas), driver='GTiff', compress='lzw')

    # 2. Calcular y escribir
    with rasterio.open(ruta_salida_archivo, 'w', **meta) as dst:
        for idx_banda, banda in enumerate(bandas, start=1):
            stack_temporal = []

            for fecha in fechas_del_mes:
                ruta_banda = os.path.join(ruta_tile, fecha, f"{tile_id}_{fecha}_{banda}.jp2")
                if os.path.exists(ruta_banda):
                    with rasterio.open(ruta_banda) as src:
                        stack_temporal.append(src.read(1))

            if stack_temporal:
                matriz_stack = np.stack(stack_temporal, axis=0)
                matriz_mediana = np.median(matriz_stack, axis=0).astype(meta['dtype'])
                dst.write(matriz_mediana, idx_banda)
            else:
                print(f"[{mes}] Advertencia: Sin datos para la banda {banda}")

    return f"[{mes}] Completado exitosamente."

def generar_composiciones_mensuales(tile_id, ruta_base_s2, dir_salida, max_workers=None):
    os.makedirs(dir_salida, exist_ok=True)
    ruta_tile = os.path.join(ruta_base_s2, tile_id)

    fechas = sorted([d for d in os.listdir(ruta_tile) if os.path.isdir(os.path.join(ruta_tile, d))])
    agrupacion_mensual = defaultdict(list)
    for fecha in fechas:
        mes = fecha[:6]
        agrupacion_mensual[mes].append(fecha)

    bandas = ["B02", "B03", "B04", "B08"]

    print(f"Procesando Tile {tile_id} | {len(agrupacion_mensual)} meses a calcular.")

    # Empaquetar los argumentos para la función paralela
    tareas = []
    for mes, fechas_del_mes in agrupacion_mensual.items():
        tareas.append((mes, fechas_del_mes, tile_id, ruta_tile, dir_salida, bandas))

    # Usar el 100% de los núcleos si max_workers es None
    nucleos_disponibles = max_workers or os.cpu_count()
    print(f"Desplegando {nucleos_disponibles} workers en la CPU...")

    # Ejecución en paralelo
    with concurrent.futures.ProcessPoolExecutor(max_workers=nucleos_disponibles) as executor:
        resultados = executor.map(procesar_un_mes, tareas)

        for resultado in resultados:
            print(resultado)

if __name__ == "__main__":
    TILE_PRUEBA = "T20JLL"
    BASE_S2 = "/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba"
    DIR_COMPOSITES = "./dataset/composites"

    # Si quieres limitar el uso de CPU (ej. a 4 núcleos), cambia max_workers=4
    generar_composiciones_mensuales(TILE_PRUEBA, BASE_S2, DIR_COMPOSITES, max_workers=None)
