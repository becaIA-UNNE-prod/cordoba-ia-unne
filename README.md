# Clasificación de Cultivos — Córdoba (UNNE)

Segmentación semántica píxel a píxel de coberturas agrícolas sobre imágenes Sentinel-2 multitemporales del tile **20JLL** (Córdoba, Argentina), período 2017-2018.

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
│   ├── 03_extraer_parches.py           # Extracción de parches 256×256 → .npz
│   ├── 1_train.py                      # Entrenamiento del modelo
│   ├── 2_infer.py                      # Inferencia con visualización (parches de test)
│   └── 2_inferencia.py                 # Inferencia sobre un tile completo → GeoTIFF
├── utils/
│   ├── model.py                        # Definición de DoubleConv y SimpleUNet
│   ├── dataset.py                      # CordobaDataset — carga parches .npz
│   └── utils.py                        # Helpers (obs2dict, dict2obj)
└── test/
    ├── 0_escanear_sentinel.py          # Exploración de archivos Sentinel-2
    └── test.py                         # Scripts de prueba
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
    │   └── etiqueta_20JLL_10m_test.tif   ← máscara reproyectada a 10 m
    ├── composites/
    │   └── 20JLL_<YYYYMM>_median.tif    ← composite mensual (4 bandas)
    ├── train/
    │   ├── dataset_20JLL.npz            ← parches X/Y + metadatos (Cnf.file_dataset)
    │   └── posiciones_parches.npy       ← coordenadas (x, y) de cada parche
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

### Paso 1 — Reproyectar etiquetas

```bash
cd src
python 01_moldear_mascaras.py
```

Toma el mapa de cobertura original (30 m, e.g. Landsat) y lo reproyecta a la resolución y grilla de Sentinel-2 (10 m) usando vecino más cercano. Salida: `etiqueta_<tile>.npz`.

---

### Paso 2 — Generar composites mensuales

```bash
cd src
python 02_temporal_median_composites.py
```

Agrupa las escenas Sentinel-2 raw por mes y calcula la **mediana temporal** por píxel para las bandas B02, B03, B04 y B08. Procesa los meses en paralelo usando todos los núcleos disponibles. Salida: un `.tif` por mes con 4 bandas.

---

### Paso 3 — Extraer parches

```bash
cd src
python 03_extraer_parches.py
```

Recorre el tile con una ventana deslizante de 256×256 px (sin solapamiento), descarta parches sin etiquetas (solo ceros) y apila todos los meses. Genera:
- `dat/train/dataset_<tile>.npz` — arrays `X` (parches imagen) e `Y` (parches etiqueta), más metadatos de meses y tile.
- `dat/train/posiciones_parches.npy` — coordenadas de cada parche, necesarias para reconstruir el mapa en inferencia.

---

### Paso 4 — Entrenar

```bash
cd src
python 1_train.py
```

Lee la configuración desde `cnf.py`. Divide el dataset en 70 / 15 / 15 (train / val / test) aleatoriamente con semilla fija. Entrena `SimpleUNet` con Adam y CrossEntropyLoss (ignorando clase 0). Aplica early stopping. Al terminar guarda en `cnf.dir_exp/`:

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
python 2_infer.py
```

Carga `best_model.pth` y reconstruye el mapa completo del tile a partir de los parches de test. Produce en `cnf.dir_exp/fig/`:

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
| `file_dataset` | `{base_dir}/train/dataset_20JLL.npz` | Ruta al dataset de parches |
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
