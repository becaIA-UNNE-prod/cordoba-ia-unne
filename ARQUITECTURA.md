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
03_extraer_parches.py
  → Recorte de parches 256×256 px con sliding window (sin overlap)
  → Descarta parches sin etiquetas (solo ceros)
  → Apila todos los meses: forma final (Meses × 4, 256, 256)
  → Salida: X_{tile}_{n}.npy  /  Y_{tile}_{n}.npy
        │
        ▼
04_split_dataset.py
  → Divide en Train / Val / Test
  → Genera split_index.json con las listas de archivos por split
        │
        ▼
1_train.py  →  SimpleUNet  →  pesos/modelo_cordoba_test.pth
```

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
| `utils/dataset.py` | `CordobaDataset` — carga y normaliza parches `.npy` |
| `src/02_temporal_median_composites.py` | Genera composites mensuales desde imágenes Sentinel-2 raw |
| `src/03_extraer_parches.py` | Extrae parches 256×256 y los guarda como `.npy` |
| `src/04_split_dataset.py` | Divide parches en train/val/test y genera el índice |
| `src/1_train.py` | Bucle de entrenamiento, validación y evaluación final |
| `src/2_inferencia.py` | Inferencia sobre nuevos composites |

---

## Rutas de datos

| Directorio | Contenido |
|---|---|
| `/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba_2017_2018/` | Imágenes Sentinel-2 raw (.jp2) organizadas por tile y fecha |
| `/mnt/yacy_1/prod/ferreyra/dataset/composites/` | Composites mensuales (.tif, 4 bandas) |
| `/mnt/yacy_1/prod/ferreyra/dataset/train/` | Parches de entrenamiento (X_*.npy, Y_*.npy) |
| `/mnt/yacy_1/prod/ferreyra/dataset/split_index.json` | Índice con listas de archivos por split |
| `./pesos/modelo_cordoba_test.pth` | Pesos guardados tras el entrenamiento |
