import rasterio

# Cambia esto por la ruta real de tu archivo TIF
ruta_mascara = "cobertura_y_uso_2021.tif" 
# Cambia esto por una de tus bandas reales de 2020
ruta_banda = "/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba/T19HGB/20200303/T19HGB_20200303_B03.jp2"

with rasterio.open(ruta_banda) as src_img:
    print("--- SENTINEL-2 ---")
    print("Dimensiones:", src_img.shape) # Debería ser algo como (10980, 10980)
    print("CRS (Proyección):", src_img.crs)
    print("Transform:", src_img.transform)

with rasterio.open(ruta_mascara) as src_mask:
    print("\n--- MÁSCARA TIF ---")
    print("Dimensiones totales:", src_mask.shape)
    print("CRS (Proyección):", src_mask.crs)
    print("Número de bandas en el tif:", src_mask.count)
