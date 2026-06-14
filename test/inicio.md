
# Estructura de directorios
El dataset sigue una estructura jerárquica estricta de tres niveles: `Ruta_Base -> Tile (MGRS) -> Fecha (YYYYMMDD) -> Archivos de Banda`.
* **Formato de Archivo:** JPEG2000 (`.jp2`)
* **Bandas Disponibles:** B02, B03, B04, B08 (Todas a 10 metros de resolución espacial).
```
/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba/
└── [TILE_MGRS]/                     # Ejemplo: T19HGB, T20JKP (Grilla Militar MGRS)
    └── [FECHA_YYYYMMDD]/            # Ejemplo: 20190826 (Año, Mes, Día)
        ├── [TILE]_[FECHA]_B02.jp2   # Banda Azul (Blue) - 10m
        ├── [TILE]_[FECHA]_B03.jp2   # Banda Verde (Green) - 10m
        ├── [TILE]_[FECHA]_B04.jp2   # Banda Roja (Red) - 10m
        └── [TILE]_[FECHA]_B08.jp2   # Banda Infrarrojo Cercano (NIR) - 10m
```
