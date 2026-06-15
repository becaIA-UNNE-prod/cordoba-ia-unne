import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import sys

# Asegurar que encuentre la carpeta utils
sys.path.append(os.path.abspath("."))
from utils.model import SimpleUNet

def generar_mapa_clasificacion():
    # 1. Configuración idéntica al entrenamiento
    TILE = "T20JLL"
    BASE_S2 = "/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba"
    RUTA_MASCARA = "./mascaras_procesadas/etiqueta_T20JLL_10m_test.tif"
    RUTA_PESOS = "./pesos/modelo_cordoba_test.pth"
    RUTA_SALIDA = "./prediccion_T20JLL.tif"
    
    IN_CHANNELS = 12
    NUM_CLASSES = 50
    SIZE = 512  # Ventana de inferencia (puede ser más grande que la de entrenamiento si la GPU lo soporta)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando inferencia en: {device}")

    # 2. Cargar el modelo entrenado
    model = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    if not os.path.exists(RUTA_PESOS):
        raise FileNotFoundError(f"No se encontraron los pesos en {RUTA_PESOS}")
        
    model.load_state_dict(torch.load(RUTA_PESOS, map_location=device))
    model.eval() # Modo evaluación (desactiva BatchNorm/Dropout)

    # 3. Preparar rutas de las imágenes de entrada
    ruta_tile = os.path.join(BASE_S2, TILE)
    fechas = sorted([d for d in os.listdir(ruta_tile) if os.path.isdir(os.path.join(ruta_tile, d))])[:3]
    bandas = ["B02", "B03", "B04", "B08"]

    # 4. Usar la máscara original solo para copiar sus metadatos espaciales (CRS, Transform)
    with rasterio.open(RUTA_MASCARA) as src_ref:
        meta = src_ref.meta.copy()
        alto, ancho = src_ref.height, src_ref.width
        # Actualizamos para asegurar que la salida sea de 1 banda entera (clases de 0 a 50)
        meta.update(dtype=rasterio.uint8, count=1, nodata=0) 

    print(f"Dimensiones del mapa a predecir: {ancho} x {alto} píxeles.")
    print("Prediciendo por bloques y escribiendo en disco...")

    # 5. Bucle de Inferencia por Ventanas
    with rasterio.open(RUTA_SALIDA, 'w', **meta) as dst:
        with torch.no_grad(): # No calculamos gradientes
            for y in range(0, alto, SIZE):
                for x in range(0, ancho, SIZE):
                    # Manejo de bordes (para no leer/escribir fuera de los límites de la imagen)
                    w = min(SIZE, ancho - x)
                    h = min(SIZE, alto - y)
                    ventana = Window(x, y, w, h)
                    
                    parche_x_temporal = []
                    
                    # Leer apilamiento temporal
                    for fecha in fechas:
                        for banda in bandas:
                            ruta_banda = os.path.join(ruta_tile, fecha, f"{TILE}_{fecha}_{banda}.jp2")
                            if os.path.exists(ruta_banda):
                                with rasterio.open(ruta_banda) as src_banda:
                                    parche = src_banda.read(1, window=ventana)
                                    parche_x_temporal.append(parche)
                            else:
                                parche_x_temporal.append(np.zeros((h, w), dtype=np.uint16))
                    
                    # Apilar y normalizar igual que en dataset.py
                    parche_x = np.stack(parche_x_temporal, axis=0).astype(np.float32)
                    parche_x = parche_x / 10000.0
                    
                    # Convertir a tensor y agregar dimensión de Batch (Batch, Channels, Height, Width)
                    tensor_x = torch.from_numpy(parche_x).unsqueeze(0).to(device)
                    
                    # Predicción de la red
                    salida = model(tensor_x)
                    
                    # Argmax para obtener la clase con mayor probabilidad por píxel
                    prediccion = torch.argmax(salida, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    
                    # Escribir el bloque predecido directamente en el TIF final
                    dst.write(prediccion, 1, window=ventana)
                    
    print(f"Inferencia completada con éxito. Mapa guardado en: {RUTA_SALIDA}")

if __name__ == "__main__":
    generar_mapa_clasificacion()
