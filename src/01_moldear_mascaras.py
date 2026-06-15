import os
import rasterio
from rasterio.warp import reproject, Resampling

def generar_mascara_10m(ruta_etiqueta_30m, ruta_referencia_s2, ruta_salida):
    print(f"Usando referencia: {ruta_referencia_s2}")
    print(f"Generando máscara en: {ruta_salida}")

    with rasterio.open(ruta_referencia_s2) as ref:
        crs_s2 = ref.crs
        transform_s2 = ref.transform
        width_s2 = ref.width
        height_s2 = ref.height

    with rasterio.open(ruta_etiqueta_30m) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs_s2,
            'transform': transform_s2,
            'width': width_s2,
            'height': height_s2
        })

        with rasterio.open(ruta_salida, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform_s2,
                dst_crs=crs_s2,
                resampling=Resampling.nearest
            )
    print("Máscara generada con éxito.")

if __name__ == "__main__":
    # Rutas ajustadas para ejecutar desde ~/cordoba_ia_unne/
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

    ruta_salida_mascara = os.path.join(DIR_SALIDA, f"etiqueta_{tile_prueba}_10m_test.tif")

    if not os.path.exists(ruta_referencia):
        raise FileNotFoundError(f"No se encontró: {ruta_referencia}")

    generar_mascara_10m(TIF_ORIGINAL, ruta_referencia, ruta_salida_mascara)
