import os
import sys
import argparse
import rasterio
import numpy as np
from collections import defaultdict
import concurrent.futures

sys.path.append(os.path.abspath("../"))
from cnf import Cnf

def procesar_un_mes(args):
    """
    Función aislada que procesa un solo mes.
    Ideal para ser ejecutada por un worker en paralelo.
    """
    mes, fechas_del_mes, tile_id, ruta_tile, dir_salida, bandas, force = args
    ruta_salida_archivo = os.path.join(dir_salida, f"{tile_id}_{mes}_median.tif")

    if os.path.exists(ruta_salida_archivo) and not force:
        return f"[{tile_id}][{mes}] Ya existe. Omitiendo (usar --force para regenerar)."

    print(f"[{tile_id}][{mes}] Iniciando cálculo con {len(fechas_del_mes)} fechas...")

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
                print(f"[{tile_id}][{mes}] Advertencia: Sin datos para la banda {banda}")

    return f"[{tile_id}][{mes}] Completado exitosamente."

def generar_composiciones_mensuales(tile_id, ruta_base_s2, dir_salida, max_workers=None, force=False):
    os.makedirs(dir_salida, exist_ok=True)
    ruta_tile = os.path.join(ruta_base_s2, tile_id)

    if not os.path.isdir(ruta_tile):
        print(f"[{tile_id}] Error: no existe {ruta_tile}, se salta este tile.")
        return

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
        tareas.append((mes, fechas_del_mes, tile_id, ruta_tile, dir_salida, bandas, force))

    # Usar el 100% de los núcleos si max_workers es None
    nucleos_disponibles = max_workers or os.cpu_count()
    print(f"Desplegando {nucleos_disponibles} workers en la CPU...")

    # Ejecución en paralelo
    with concurrent.futures.ProcessPoolExecutor(max_workers=nucleos_disponibles) as executor:
        resultados = executor.map(procesar_un_mes, tareas)

        for resultado in resultados:
            print(resultado)

if __name__ == "__main__":
    cnf = Cnf()

    parser = argparse.ArgumentParser(
        description="Paso 2: calcula composites medianos mensuales (B02,B03,B04,B08) desde "
                    "imagenes Sentinel-2 raw, de uno o varios tiles. Salida: "
                    "{dir_salida}/<tile>_<YYYYMM>_median.tif",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tiles", nargs="+", default=cnf.tiles,
                        help="Tiles a procesar. Si no se pasa, usa Cnf.tiles")
    parser.add_argument("--base-s2", default="/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba",
                        help="Directorio raiz de las imagenes Sentinel-2 raw")
    parser.add_argument("--dir-salida", default=f"{cnf.base_dir}/composites",
                        help="Directorio de salida de los composites")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Nucleos de CPU a usar en paralelo (default: todos los disponibles)")
    parser.add_argument("--force", action="store_true",
                        help="Recalcula el composite de un mes aunque el .tif ya exista")
    args = parser.parse_args()

    # Corriendo esto de nuevo con --tiles ampliado (o Cnf.tiles ampliado y sin
    # --tiles) solo calcula lo que falte: cada mes ya existente se saltea
    # (por tile y por mes), salvo que se pase --force.
    for tile_id in args.tiles:
        generar_composiciones_mensuales(tile_id, args.base_s2, args.dir_salida,
                                        max_workers=args.max_workers, force=args.force)
