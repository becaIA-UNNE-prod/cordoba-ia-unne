import os
import sys
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import time

sys.path.append(os.path.abspath("../"))
from cnf import Cnf


def extraer_parches_desde_composites(tile_id, ruta_mascara, dir_composites, dir_salida, size=256, step=None):
    """
    Extrae parches de un unico tile y los guarda como:
        X_<tile_id>.npy       (N, meses*bandas, size, size) float32
        Y_<tile_id>.npy       (N, size, size)               int64
        meta_<tile_id>.npz    meses, tile_id, n_samples (metadata liviana)
        posiciones_<tile_id>.npy  (N, 2) coordenadas (x, y) de cada parche

    X/Y se guardan sin comprimir (np.save, no np.savez_compressed) para que
    utils/dataset.py pueda abrirlos con mmap_mode='r' y entrenar con varios
    tiles sin cargarlos todos a RAM. Ver utils/dataset.py para el detalle.

    Las posiciones se arman en el mismo recorrido que descarta los parches
    sin etiquetas, para que posiciones_<tile_id>.npy[i] corresponda siempre
    a X_<tile_id>.npy[i] / Y_<tile_id>.npy[i] (antes se generaban por
    separado asumiendo que ningun parche se descartaba, lo cual desalineaba
    los indices apenas habia parches vacios).
    """
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
    posiciones = []

    # Cargar mascara desde .npz o .tif
    if ruta_mascara.endswith('.npz'):
        data = np.load(ruta_mascara, allow_pickle=True)
        etiquetas_full = data['etiquetas']
        alto, ancho = etiquetas_full.shape
    else:
        with rasterio.open(ruta_mascara) as src_label:
            etiquetas_full = src_label.read(1)
            alto, ancho = src_label.height, src_label.width

    contador_parches = 0
    n_parches_x = (ancho - size) // step + 1
    n_parches_y = (alto - size) // step + 1
    total_posibles = n_parches_x * n_parches_y

    print(f"[{tile_id}] Recorriendo {n_parches_x}x{n_parches_y} = {total_posibles:,} parches...")

    start_time = time.time()

    for y in range(0, alto - size, step):
        for x in range(0, ancho - size, step):
            ventana = Window(x, y, size, size)
            parche_y = etiquetas_full[y:y+size, x:x+size]

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
            posiciones.append((x, y))
            contador_parches += 1
            if contador_parches % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[{tile_id}] Procesados {contador_parches} parches en {elapsed:.1f}s...")

    if contador_parches == 0:
        print(f"[{tile_id}] Advertencia: No se encontraron parches con cultivos")
        return None

    X = np.stack(parches_x, axis=0)
    Y = np.stack(parches_y, axis=0)

    os.makedirs(dir_salida, exist_ok=True)
    paths = Cnf.paths_tile(tile_id, dir_dataset=dir_salida)

    np.save(paths["x"], X)
    np.save(paths["y"], Y)
    np.savez(paths["meta"], meses=archivos_mensuales, tile_id=tile_id, n_samples=contador_parches)
    np.save(paths["posiciones"], np.array(posiciones, dtype=np.int64))

    print(f"[{tile_id}] Guardados {contador_parches} parches:")
    print(f"  {paths['x']}")
    print(f"  {paths['y']}")
    print(f"  {paths['meta']}")
    print(f"  {paths['posiciones']}")
    return paths


def cargar_dataset_tile(tile_id, dir_dataset):
    """ Carga a memoria un tile ya extraido (util para inspeccionar / debug, no para entrenar). """
    paths = Cnf.paths_tile(tile_id, dir_dataset=dir_dataset)
    X = np.load(paths["x"])
    Y = np.load(paths["y"])
    meta = np.load(paths["meta"], allow_pickle=True)
    return X, Y, meta['meses'], str(meta['tile_id'])


if __name__ == "__main__":
    cnf = Cnf()

    parser = argparse.ArgumentParser(
        description="Paso 3: extrae parches 256x256 (sliding window, sin overlap) de uno o varios "
                    "tiles ya compuestos y los guarda en formato mmap-eable (.npy) listo para "
                    "entrenar. Salida por tile: X_<tile>.npy, Y_<tile>.npy, meta_<tile>.npz, "
                    "posiciones_<tile>.npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tiles", nargs="+", default=cnf.tiles,
                        help="Tiles a extraer. Si no se pasa, usa Cnf.tiles")
    parser.add_argument("--etiquetas-dir", default=f"{cnf.base_dir}/etiquetas",
                        help="Directorio con etiqueta_<tile>.npz por tile")
    parser.add_argument("--composites-dir", default=f"{cnf.base_dir}/composites",
                        help="Directorio con los composites mensuales por tile")
    parser.add_argument("--dir-salida", default=cnf.dir_dataset,
                        help="Directorio de salida de los parches (Cnf.dir_dataset)")
    parser.add_argument("--size", type=int, default=cnf.size_parche,
                        help="Tamano de parche en px")
    parser.add_argument("--step", type=int, default=cnf.step_parche,
                        help="Paso del sliding window en px")
    parser.add_argument("--force", action="store_true",
                        help="Reextrae un tile aunque X_<tile>.npy ya exista")
    args = parser.parse_args()

    # Corriendo esto de nuevo con --tiles ampliado (o Cnf.tiles ampliado y sin
    # --tiles) solo extrae los tiles nuevos: los que ya tienen X_<tile>.npy se
    # saltean salvo que se pase --force.
    for tile_id in args.tiles:
        paths_tile = Cnf.paths_tile(tile_id, dir_dataset=args.dir_salida)

        if os.path.exists(paths_tile["x"]) and not args.force:
            print(f"[{tile_id}] Ya existe {paths_tile['x']}, se salta (usar --force para reextraer).")
            continue

        ruta_mascara = os.path.join(args.etiquetas_dir, f"etiqueta_{tile_id}.npz")
        if not os.path.exists(ruta_mascara):
            print(f"[{tile_id}] Error: no se encontro la mascara {ruta_mascara}, se salta este tile.")
            continue

        resultado = extraer_parches_desde_composites(
            tile_id,
            ruta_mascara,
            args.composites_dir,
            args.dir_salida,
            size=args.size,
            step=args.step,
        )

        if resultado:
            X, Y, meses, tile = cargar_dataset_tile(tile_id, args.dir_salida)
            print(f"[{tile}] X shape: {X.shape} | Y shape: {Y.shape} | Meses: {len(meses)}")
