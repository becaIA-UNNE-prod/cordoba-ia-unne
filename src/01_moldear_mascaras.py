import os
import sys
import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.append(os.path.abspath("../"))
from cnf import Cnf

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

def encontrar_banda_referencia(tile_id, base_s2, banda="B04"):
    """
    Busca la primera fecha disponible de un tile que tenga la banda pedida,
    para usarla como grilla de referencia (CRS, transform, tamaño) al
    reproyectar las etiquetas a ese tile.
    """
    ruta_tile = os.path.join(base_s2, tile_id)
    if not os.path.isdir(ruta_tile):
        raise FileNotFoundError(f"No existe el directorio del tile: {ruta_tile}")

    fechas = sorted(f for f in os.listdir(ruta_tile) if os.path.isdir(os.path.join(ruta_tile, f)))
    for fecha in fechas:
        candidato = os.path.join(ruta_tile, fecha, f"{tile_id}_{fecha}_{banda}.jp2")
        if os.path.exists(candidato):
            return candidato

    raise FileNotFoundError(
        f"No se encontro ninguna fecha con banda {banda} para el tile {tile_id} en {ruta_tile}"
    )


def procesar_tile(tile_id, ruta_etiqueta_30m, base_s2, dir_salida, force=False):
    ruta_salida_npz = os.path.join(dir_salida, f"etiqueta_{tile_id}.npz")

    if os.path.exists(ruta_salida_npz) and not force:
        print(f"[{tile_id}] Ya existe {ruta_salida_npz}, se salta (usar --force para regenerar).")
        return ruta_salida_npz

    ruta_referencia = encontrar_banda_referencia(tile_id, base_s2)
    print(f"[{tile_id}] Usando como referencia de grilla: {ruta_referencia}")

    generar_etiquetas_como_npz(ruta_etiqueta_30m, ruta_referencia, ruta_salida_npz)
    return ruta_salida_npz


if __name__ == "__main__":
    cnf = Cnf()

    parser = argparse.ArgumentParser(
        description="Paso 1: reproyecta el mapa de coberturas original (30m) a la grilla "
                    "Sentinel-2 (10m) de uno o varios tiles. Salida: "
                    "{base_dir}/etiquetas/etiqueta_<tile>.npz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tiles", nargs="+", default=cnf.tiles,
                        help="Tiles a procesar. Si no se pasa, usa Cnf.tiles")
    parser.add_argument("--origen", default="./Nivel3_28_dic_2018_30m_completo.tif",
                        help="Ruta al raster de coberturas original (30m)")
    parser.add_argument("--base-s2", default="/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba",
                        help="Directorio raiz de las imagenes Sentinel-2 raw, organizado por tile/fecha")
    parser.add_argument("--force", action="store_true",
                        help="Regenera la etiqueta aunque el .npz de salida ya exista")
    args = parser.parse_args()

    if not os.path.exists(args.origen):
        raise FileNotFoundError(f"No se encontró el raster de origen: {args.origen}")

    dir_salida = f"{cnf.base_dir}/etiquetas"
    os.makedirs(dir_salida, exist_ok=True)

    # Corriendo esto de nuevo con --tiles ampliado (o Cnf.tiles ampliado y sin
    # --tiles) solo procesa los tiles nuevos: los que ya tienen su
    # etiqueta_<tile>.npz se saltean salvo que se pase --force.
    for tile_id in args.tiles:
        try:
            procesar_tile(tile_id, args.origen, args.base_s2, dir_salida, force=args.force)
        except FileNotFoundError as e:
            print(f"[{tile_id}] Error: {e}. Se salta este tile.")
