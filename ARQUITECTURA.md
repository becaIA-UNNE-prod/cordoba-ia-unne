# Arquitectura del modelo — Clasificación de cultivos Córdoba (UNNE)

## Objetivo

Segmentación semántica píxel a píxel de coberturas agrícolas sobre imágenes Sentinel-2 multitemporales del tile **20JLL** (Córdoba, Argentina), usando información de los años 2017-2018.

---

## Pipeline de datos

```
Sentinel-2 (raw .jp2)
        │
        ▼
02_temporal_median_composites.py
  → Composición mediana mensual por tile y banda
  → Salida: {tile}_{YYYYMM}_median.tif  (4 bandas: B02, B03, B04, B08)
        │
        ▼
03_extraer_parches.py  (uno o varios tiles, ver Cnf.tiles)
  → Recorte de parches 256×256 px con sliding window (sin overlap)
  → Descarta parches sin etiquetas (solo ceros)
  → Apila todos los meses: forma final (Meses × 4, 256, 256)
  → Salida por tile (sin comprimir, para poder abrir con mmap):
      X_{tile}.npy, Y_{tile}.npy, meta_{tile}.npz, posiciones_{tile}.npy
        │
        ▼
1_train.py
  → CordobaDataset abre todos los tiles de Cnf.tiles con mmap_mode='r'
    (RAM acotada al tamaño de un batch, no al tamaño de los tiles)
  → Split aleatorio Train / Val / Test (70/15/15) sobre el dataset combinado
  → SimpleUNet  →  {dir_exp}/best_model.pth, modelo_final.pth
```

**Por qué `.npy` y no `.npz` para X/Y:** un `.npz` (comprimido o no) no soporta memory-mapping en numpy — acceder a `data['X']` siempre descomprime el array completo a RAM. Con un solo tile chico eso entraba en memoria sin problema; para entrenar con varios tiles a la vez hace falta que cada `__getitem__` traiga a RAM solo el parche pedido, y eso solo lo permite un `.npy` sin comprimir abierto con `mmap_mode='r'`. `meta_{tile}.npz` sigue siendo `.npz` porque solo guarda strings (meses, tile_id), es chico y no necesita mmap.

---

## Entrada al modelo

| Propiedad | Valor |
|---|---|
| Formato de parche | `.npy` (NumPy float32) |
| Dimensiones espaciales | 256 × 256 px |
| Canales | `Meses × 4` (autodetectado al inicio del entrenamiento) |
| Bandas por mes | B02 (azul), B03 (verde), B04 (rojo), B08 (NIR) |
| Normalización | `x / 10000.0` (reflectancia superficial Sentinel-2 L2A) |
| Etiquetas (`Y`) | Mapa entero int64 de clases por píxel |
| Clase 0 | NoData / fondo — ignorada en loss y métricas |

---

## Arquitectura: SimpleUNet

Variante simplificada de **U-Net** con 2 niveles de profundidad.

### Bloque base — DoubleConv

```
Conv2d(in→out, 3×3, padding=1)
BatchNorm2d(out)
ReLU
Conv2d(out→out, 3×3, padding=1)
BatchNorm2d(out)
ReLU
```

Mantiene las dimensiones espaciales intactas (same-padding).

---

### Diagrama completo

```
Entrada: (B, C_in, 256, 256)
          │
          ▼
┌─────────────────────────────┐
│  ENCODER                    │
│                             │
│  DoubleConv(C_in → 64)  ───────────────────── skip c1
│  MaxPool2d(2)               │                    │
│                             │                    │
│  DoubleConv(64 → 128)   ───────────────── skip c2│
│  MaxPool2d(2)               │                │   │
└─────────────────────────────┘                │   │
                │                              │   │
                ▼                              │   │
┌─────────────────────────────┐               │   │
│  CUELLO DE BOTELLA          │               │   │
│  DoubleConv(128 → 256)      │               │   │
└─────────────────────────────┘               │   │
                │                              │   │
                ▼                              │   │
┌─────────────────────────────┐               │   │
│  DECODER                    │               │   │
│                             │               │   │
│  ConvTranspose2d(256→128, 2×2)              │   │
│  Concat(↑, c2)  ←──────────────────────────┘   │
│  DoubleConv(256 → 128)      │                   │
│                             │                   │
│  ConvTranspose2d(128→64, 2×2)                   │
│  Concat(↑, c1)  ←──────────────────────────────┘
│  DoubleConv(128 → 64)       │
└─────────────────────────────┘
                │
                ▼
        Conv2d(64 → num_classes, 1×1)
                │
                ▼
      Salida: (B, num_classes, 256, 256)
```

---

### Tabla de capas

| Bloque | Capa | Entrada | Salida | Parámetros clave |
|---|---|---|---|---|
| Encoder | DoubleConv | (B, C_in, 256, 256) | (B, 64, 256, 256) | 3×3, padding=1 |
| Encoder | MaxPool2d | (B, 64, 256, 256) | (B, 64, 128, 128) | kernel=2 |
| Encoder | DoubleConv | (B, 64, 128, 128) | (B, 128, 128, 128) | 3×3, padding=1 |
| Encoder | MaxPool2d | (B, 128, 128, 128) | (B, 128, 64, 64) | kernel=2 |
| Bottleneck | DoubleConv | (B, 128, 64, 64) | (B, 256, 64, 64) | 3×3, padding=1 |
| Decoder | ConvTranspose2d | (B, 256, 64, 64) | (B, 128, 128, 128) | 2×2, stride=2 |
| Decoder | Concat (skip c2) | — | (B, 256, 128, 128) | — |
| Decoder | DoubleConv | (B, 256, 128, 128) | (B, 128, 128, 128) | 3×3, padding=1 |
| Decoder | ConvTranspose2d | (B, 128, 128, 128) | (B, 64, 256, 256) | 2×2, stride=2 |
| Decoder | Concat (skip c1) | — | (B, 128, 256, 256) | — |
| Decoder | DoubleConv | (B, 128, 256, 256) | (B, 64, 256, 256) | 3×3, padding=1 |
| Salida | Conv2d | (B, 64, 256, 256) | (B, num_classes, 256, 256) | 1×1 |

Las **skip connections** concatenan feature maps del encoder con los del decoder en el mismo nivel espacial, preservando detalles de alta resolución.

---

## Configuración de entrenamiento

| Hiperparámetro | Valor |
|---|---|
| `num_classes` | 50 |
| `batch_size` | 4 |
| `epochs` | 5 |
| `learning_rate` | 1e-4 |
| Optimizador | Adam |
| Función de loss | CrossEntropyLoss (`ignore_index=0`) |
| Dispositivo | CUDA si disponible, CPU si no |
| Métrica | Pixel accuracy (excluyendo clase 0) |

---

## Archivos del proyecto

| Archivo | Rol |
|---|---|
| `utils/model.py` | Definición de `DoubleConv` y `SimpleUNet` |
| `utils/dataset.py` | `CordobaDataset` — dataset multi-tile, abre `X_<tile>.npy`/`Y_<tile>.npy` con mmap, normaliza y remapea etiquetas |
| `src/01_moldear_mascaras.py` | Reproyecta la máscara de etiquetas a la grilla Sentinel-2, por tile |
| `src/02_temporal_median_composites.py` | Genera composites mensuales desde imágenes Sentinel-2 raw, por tile |
| `src/03_extraer_parches.py` | Extrae parches 256×256 por tile y los guarda como `.npy` (mmap-eables) + metadata |
| `src/1_train.py` | Bucle de entrenamiento (sobre `Cnf.tiles`), validación y evaluación final |
| `src/2_infer.py` | Inferencia: `--modo test` (parches de test, por tile) o `--modo mapa` (GeoTIFF de un tile completo) |
| `test/test_dataset.py` | Test de humo de `CordobaDataset` con tiles sintéticos, sin depender del servidor |

---

## Rutas de datos

| Directorio | Contenido |
|---|---|
| `/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba_2017_2018/` | Imágenes Sentinel-2 raw (.jp2) organizadas por tile y fecha |
| `/mnt/yacy_1/prod/ferreyra/dataset/etiquetas/` | Máscaras reproyectadas (`etiqueta_<tile>.npz`), una por tile |
| `/mnt/yacy_1/prod/ferreyra/dataset/composites/` | Composites mensuales (.tif, 4 bandas), por tile y mes |
| `/mnt/yacy_1/prod/ferreyra/dataset/train/` | Parches de entrenamiento: `X_<tile>.npy`, `Y_<tile>.npy`, `meta_<tile>.npz`, `posiciones_<tile>.npy` |
| `{Cnf.dir_exp}/` | Pesos y métricas del experimento (`best_model.pth`, `modelo_final.pth`, `metricas_entrenamiento.npz`, `config.txt`) |
