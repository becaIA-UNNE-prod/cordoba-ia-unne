# Clasificación de Cultivos — Córdoba (UNNE)

Segmentación semántica píxel a píxel de coberturas agrícolas sobre imágenes Sentinel-2 multitemporales de Córdoba (Argentina), período 2017-2018. El pipeline arrancó con el tile **T20JLL** pero el dataset y el entrenamiento soportan combinar varios tiles (ver `Cnf.tiles` en Configuración).

---

## Clases del modelo

El raster de etiquetas original (Nivel3) tiene 28 valores (0–27). Al cargar el dataset se aplica un remapeo que los reduce a **8 clases**:

| ID nuevo | Clase | IDs originales |
|----------|-------|----------------|
| 0 | NoData — ignorado en entrenamiento y métricas | 0 |
| 1 | No cultivos (monte, pastizales, agua, urbano, vial…) | 1–14 |
| 2 | Trigo | 15 |
| 3 | Maíz | 16 |
| 4 | Soja | 17 |
| 5 | Maní | 18 |
| 6 | Sorgo | 19 |
| 7 | Otros cultivos (dobles cultivos, pasturas, plantaciones, frutales) | 20–27 |

El remapeo se define en `src/cnf.py` (`Cnf.label_remap`) y se aplica en `utils/dataset.py` al construir `CordobaDataset`.

---

## Estructura del proyecto

```
.
├── src/
│   ├── cnf.py                          # Configuración centralizada (hiperparámetros, rutas)
│   ├── 01_moldear_mascaras.py          # Reproyección de etiquetas a la grilla Sentinel-2
│   ├── 02_temporal_median_composites.py # Composites medianas mensuales desde .jp2 raw
│   ├── 03_extraer_parches.py           # Extracción de parches 256×256 → .npy (uno o varios tiles)
│   ├── 1_train.py                      # Entrenamiento del modelo (uno o varios tiles)
│   └── 2_infer.py                      # Inferencia: --modo test (parches de test) o --modo mapa (GeoTIFF de un tile completo)
├── utils/
│   ├── model.py                        # Definición de DoubleConv y SimpleUNet
│   ├── dataset.py                      # CordobaDataset — dataset multi-tile, carga parches .npy vía mmap
│   └── utils.py                        # Helpers (obs2dict, dict2obj)
└── test/
    ├── 0_escanear_sentinel.py          # Exploración de archivos Sentinel-2
    ├── test.py                         # Scripts de prueba
    └── test_dataset.py                 # Test de humo de CordobaDataset con tiles sintéticos (sin datos reales)
```

### Datos (servidor remoto vía SSH)

Toda la data vive en `/mnt/yacy_1/prod/ferreyra/`. La raíz configurable del proyecto es `dataset/`; la ruta base se define en `Cnf.base_dir` dentro de `src/cnf.py`.

```
/mnt/yacy_1/prod/ferreyra/
│
├── sentinel2_cordoba_2017_2018/
│   └── <tile>/
│       └── <fecha>/          ← imágenes raw .jp2 por banda
│
└── dataset/                  ← Cnf.base_dir
    ├── etiquetas/
    │   └── etiqueta_<tile>.npz           ← máscara reproyectada a 10 m, una por tile
    ├── composites/
    │   └── <tile>_<YYYYMM>_median.tif    ← composite mensual (4 bandas), uno por tile y mes
    ├── train/                            ← Cnf.dir_dataset — un set de 4 archivos por tile
    │   ├── X_<tile>.npy                  ← parches de imagen (N, meses×4, 256, 256) float32
    │   ├── Y_<tile>.npy                  ← parches de etiqueta (N, 256, 256) int64
    │   ├── meta_<tile>.npz               ← metadata liviana (meses, tile_id, n_samples)
    │   └── posiciones_<tile>.npy         ← coordenadas (x, y) de cada parche de ese tile
    └── exp<N>/                          ← Cnf.dir_exp, se crea al entrenar
        ├── best_model.pth
        ├── modelo_final.pth
        ├── test_indices.npy
        ├── metricas_entrenamiento.npz
        ├── config.txt
        └── fig/
            ├── metricas_entrenamiento.png
            ├── mapa_testing.png
            └── matriz_confusion.png
```

Para cambiar el servidor o el directorio base basta con modificar `Cnf.base_dir` en `src/cnf.py`. Todos los demás paths se derivan de él automáticamente.

---

## Pipeline

Los pasos 1 a 3 (preparación de datos) aceptan `--tiles` para elegir con qué tiles trabajar en esa corrida, en vez de depender únicamente de `Cnf.tiles`. Los tres son **idempotentes**: si el archivo de salida de un tile ya existe, lo saltean (a menos que se pase `--force`). Esto significa que agregar tiles al proyecto es simplemente volver a correr los tres pasos con la lista ampliada — los tiles ya procesados no se recalculan. Todos soportan `--help` con el detalle de cada opción.

### Paso 1 — Reproyectar etiquetas

```bash
cd src
python 01_moldear_mascaras.py --help
python 01_moldear_mascaras.py --origen ./Nivel3_28_dic_2018_30m_completo.tif --tiles T20JLL
```

Toma el mapa de cobertura original (30 m, e.g. Landsat) y lo reproyecta a la resolución y grilla de Sentinel-2 (10 m) usando vecino más cercano, para cada tile pedido (busca automáticamente la primera fecha disponible de ese tile en `--base-s2` para usar como referencia de grilla). Salida: `{base_dir}/etiquetas/etiqueta_<tile>.npz`.

| Opción | Default | Descripción |
|---|---|---|
| `--tiles` | `Cnf.tiles` | Uno o más tiles, ej. `--tiles T20JLL T19HGB` |
| `--origen` | `./Nivel3_28_dic_2018_30m_completo.tif` | Raster de coberturas original (30 m) |
| `--base-s2` | `/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba` | Raíz de las imágenes Sentinel-2 raw |
| `--force` | `False` | Regenera aunque `etiqueta_<tile>.npz` ya exista |

---

### Paso 2 — Generar composites mensuales

```bash
cd src
python 02_temporal_median_composites.py --help
python 02_temporal_median_composites.py --tiles T20JLL T19HGB
```

Agrupa las escenas Sentinel-2 raw por mes y calcula la **mediana temporal** por píxel para las bandas B02, B03, B04 y B08, para cada tile pedido. Procesa los meses en paralelo usando todos los núcleos disponibles. Salida: `<tile>_<YYYYMM>_median.tif` (un `.tif` por mes y tile).

| Opción | Default | Descripción |
|---|---|---|
| `--tiles` | `Cnf.tiles` | Uno o más tiles |
| `--base-s2` | `/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba` | Raíz de las imágenes Sentinel-2 raw |
| `--dir-salida` | `{base_dir}/composites` | Directorio de salida |
| `--max-workers` | todos los núcleos | Núcleos de CPU a usar en paralelo |
| `--force` | `False` | Recalcula un mes aunque su `.tif` ya exista |

---

### Paso 3 — Extraer parches

```bash
cd src
python 03_extraer_parches.py --help
python 03_extraer_parches.py --tiles T20JLL T19HGB
```

Recorre cada tile pedido con una ventana deslizante de 256×256 px (sin solapamiento), descarta parches sin etiquetas (solo ceros) y apila todos los meses. Por cada tile genera, en `--dir-salida` (default `Cnf.dir_dataset`):
- `X_<tile>.npy` / `Y_<tile>.npy` — parches de imagen y de etiqueta, sin comprimir.
- `meta_<tile>.npz` — metadata liviana (meses, tile_id, cantidad de parches).
- `posiciones_<tile>.npy` — coordenadas (x, y) de cada parche de ese tile, necesarias para reconstruir el mapa en inferencia.

| Opción | Default | Descripción |
|---|---|---|
| `--tiles` | `Cnf.tiles` | Uno o más tiles |
| `--etiquetas-dir` | `{base_dir}/etiquetas` | Directorio con `etiqueta_<tile>.npz` |
| `--composites-dir` | `{base_dir}/composites` | Directorio con los composites mensuales |
| `--dir-salida` | `Cnf.dir_dataset` | Directorio de salida de los parches |
| `--size` / `--step` | `Cnf.size_parche` / `Cnf.step_parche` | Tamaño de parche y paso del sliding window |
| `--force` | `False` | Reextrae aunque `X_<tile>.npy` ya exista |

**Por qué `.npy` sin comprimir y no `.npz` comprimido para X/Y:** numpy permite abrir un `.npy` con `mmap_mode='r'`, de forma que el sistema operativo pagina a RAM solo los parches que se piden en cada `__getitem__`, sin importar cuántos tiles se agreguen. Un `.npz` (comprimido o no) no soporta esto — acceder a `data['X']` siempre descomprime/copia el array **completo** a memoria. Con un solo tile chico esto pasaba desapercibido; con varios tiles cargados a la vez agotaría la RAM del servidor. `meta_<tile>.npz` sigue siendo `.npz` porque es chico (solo strings) y no necesita mmap.

---

### Agregar tiles a un proyecto ya en marcha

Para sumar un tile nuevo al dataset de entrenamiento sin retocar los que ya estaban procesados:

```bash
cd src
python 01_moldear_mascaras.py            --tiles T19HGB
python 02_temporal_median_composites.py  --tiles T19HGB
python 03_extraer_parches.py             --tiles T19HGB
```

y por último agregar `"T19HGB"` a la lista `Cnf.tiles` en `src/cnf.py` (usada por `1_train.py`/`2_infer.py`). Como los tres scripts son idempotentes, correrlos sin `--tiles` (o con la lista completa, tiles viejos + nuevo) también funciona: los tiles ya generados se saltean automáticamente y solo se procesa el nuevo — usar `--force` únicamente si se quiere regenerar algo desde cero.

Todos los tiles en `Cnf.tiles` deben cubrir la misma cantidad de meses (mismos canales de entrada); si no, `1_train.py` falla con un error explícito al construir el dataset en lugar de entrenar con datos inconsistentes.

---

### Paso 4 — Entrenar

```bash
cd src
python 1_train.py
```

Lee la configuración desde `cnf.py`, incluida la lista `Cnf.tiles` (uno o varios). Divide el dataset combinado en 70 / 15 / 15 (train / val / test) aleatoriamente con semilla fija. Entrena `SimpleUNet` con Adam y CrossEntropyLoss (ignorando clase 0). Aplica early stopping. Al terminar guarda en `cnf.dir_exp/`:

| Archivo | Descripción |
|---|---|
| `best_model.pth` | Pesos con menor val loss |
| `modelo_final.pth` | Pesos al finalizar el bucle |
| `test_indices.npy` | Índices del split de test (para inferencia reproducible) |
| `metricas_entrenamiento.npz` | Curvas de loss y accuracy por época |
| `fig/metricas_entrenamiento.png` | Gráfico de curvas de entrenamiento |
| `config.txt` | Snapshot de la configuración usada |

---

### Paso 5 — Inferencia y visualización

```bash
cd src
python 2_infer.py --tile T20JLL   # --tile es opcional, default: el primero de Cnf.tiles
```

Carga `best_model.pth` y reconstruye el mapa completo **de un tile** a partir de los parches de test que le correspondan (si el modelo se entrenó con varios tiles, `test_indices.npy` mezcla parches de todos; `--tile` elige cuál reconstruir, porque cada tile es una grilla geográfica distinta y no se pueden mezclar en un mismo lienzo). Produce en `cnf.dir_exp/fig/`:

- **`mapa_testing.png`** — Tres paneles: ground truth, predicción y mapa de errores (verde = correcto, rojo = error).
- **`matriz_confusion.png`** — Matriz de confusión normalizada para las clases agrícolas de interés (clases 2–7 del esquema remapeado).

#### Inferencia sobre un tile completo (GeoTIFF)

```bash
cd src
python 2_inferencia.py
```

Lee los composites mensuales directamente (sin dataset preempaquetado) y predice bloque a bloque sobre el tile completo (10 980 × 10 980 px). Escribe el resultado en `prediccion_T20JLL.tif`.

---

## Configuración

Toda la configuración vive en `src/cnf.py`:

| Parámetro | Valor | Descripción |
|---|---|---|
| `base_dir` | `/mnt/yacy_1/prod/ferreyra/dataset` | Raíz de todos los datos del proyecto |
| `dir_dataset` | `{base_dir}/train` | Directorio con los `X_<tile>.npy` / `Y_<tile>.npy` / `meta_<tile>.npz` / `posiciones_<tile>.npy` de cada tile |
| `tiles` | `["T20JLL"]` | Lista de tiles a usar para entrenar. Agregar más tiles (ya extraídos) para entrenar con varios |
| `dir_exp` | `{base_dir}/exp6` | Directorio de salida del experimento |
| `label_remap` | array 28 elementos | Remapeo de clases originales (0–27) a esquema de 8 clases |
| `num_classes` | 8 | Clases de salida del modelo |
| `batch_size` | 16 | Tamaño de batch |
| `epochs` | 100 | Épocas máximas |
| `learning_rate` | 5e-4 | Tasa de aprendizaje (Adam) |
| `patience` | 50 | Épocas sin mejora antes de early stopping |
| `seed` | 42 | Semilla para reproducibilidad |
| `size_parche` | 256 | Tamaño de parche en píxeles |
| `normalizar` | `True` | Divide los valores por 10 000 (reflectancia L2A) |

Para cambiar el servidor o directorio base, modificar `Cnf.base_dir`. Para cambiar de experimento sin sobrescribir resultados anteriores, incrementar el número en `dir_exp` (e.g. `exp7`).

---

## Modelo — SimpleUNet

U-Net simplificada con 2 niveles de encoder/decoder. Los canales de entrada se autodetectan al inicio del entrenamiento según la cantidad de meses disponibles (`Meses × 4 bandas`). Ver [ARQUITECTURA.md](ARQUITECTURA.md) para el diagrama completo.
