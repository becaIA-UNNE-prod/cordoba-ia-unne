import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath("../"))
from cnf import Cnf # configuracion
from utils.model import SimpleUNet
from utils.dataset import CordobaDataset

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

def visualizar_comparacion(target, prediccion, dir_exp, ignore_class=0):
    """
    Visualiza target, predicción y diferencias en 3 paneles usando un enfoque lógico
    de Acierto (True) vs. Error (False) para las zonas agrícolas de test.
    """
    # Asegurar arrays de NumPy
    if hasattr(target, 'cpu'): target = target.cpu().numpy()
    if hasattr(prediccion, 'cpu'): prediccion = prediccion.cpu().numpy()
    
    # Identificar las zonas que corresponden a parches de test (donde no es -1)
    mascara_test_real = target != -1
    
    # --- CONSTRUCCIÓN DEL MAPA DE DIFERENCIAS SIMPLIFICADO ---
    # Inicializamos TODO el mapa con 3 (Zona ajena al test / Train / Val)
    diferencias = np.full_like(target, fill_value=3, dtype=np.int8)
    
    # 1. Identificar zonas de cultivos y fondo dentro de test
    mascara_target_cultivo = (target != ignore_class) & mascara_test_real
    mascara_pred_cultivo = (prediccion != ignore_class) & mascara_test_real
    
    # [Clase 0] ACUERDO / CORRECTO (Verde): Coinciden exactamente
    # Esto incluye tanto aciertos en cultivos como aciertos en fondo dentro del test
    mascara_correctos = (target == prediccion) & mascara_test_real
    diferencias[mascara_correctos] = 0
    
    # [Clase 1] ERROR / INCORRECTO (Rojo): No coinciden en absoluto
    mascara_errores = (target != prediccion) & mascara_test_real
    diferencias[mascara_errores] = 1
    
    # [Clase 2] FONDO DE TEST PURO (Gris Claro): Opcional, para mantener la silueta
    # Si preferís que el fondo correcto también se pinte de verde (Acierto), borrá las siguientes 2 líneas
    mascara_fondo_ok = (target == ignore_class) & (prediccion == ignore_class) & mascara_test_real
    diferencias[mascara_fondo_ok] = 2

    # --- Configuración del Colormap de Clases (Paneles 1 y 2) ---
    max_clase = int(max(target.max(), prediccion.max()))
    if max_clase <= 20:
        cmap_clases = plt.cm.tab20
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20))])
        cmap_clases = ListedColormap(colors)
    
    norm_clases = BoundaryNorm(np.arange(-0.5, max_clase + 1.5, 1), cmap_clases.N)
    
    # --- RENDERIZADO DE PANELES ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Target
    axes[0].imshow(target, cmap=cmap_clases, norm=norm_clases, interpolation='none')
    axes[0].set_title('Target (Ground Truth)', fontsize=14)
    axes[0].axis('off')
    
    # Panel 2: Predicción
    axes[1].imshow(prediccion, cmap=cmap_clases, norm=norm_clases, interpolation='none')
    axes[1].set_title('Inferencia (Predicción)', fontsize=14)
    axes[1].axis('off')
    
    # Panel 3: Diferencias Lógicas (True/False)
    # Índices:      0=Verde,      1=Rojo,         2=Gris Claro,  3=Azul Oscuro
    # Significado:  Correcto,     Error (Discrepa), Fondo OK,     No es Test
    colors_diff = ['#2ecc71', '#e74c3c', '#bdc3c7', '#2c3e50']
    cmap_diff = ListedColormap(colors_diff)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_diff = BoundaryNorm(bounds, cmap_diff.N)
    
    axes[2].imshow(diferencias, cmap=cmap_diff, norm=norm_diff, interpolation='none')
    axes[2].set_title('Mapa de Errores en Test\n(Verde=Acierto, Rojo=Error, Gris=Fondo Test, Azul=No Test)', fontsize=11)
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(dir_exp, exist_ok=True)
    plt.savefig(f"{dir_exp}/fig/mapa_testing.png", dpi=300, bbox_inches='tight')
    print(f"Comparación guardada en {dir_exp}/fig/mapa_testing.png")
    plt.close()
    
    # --- Métricas globales de la imagen ---
    # Calculamos la precisión enfocado solo en las zonas donde debería haber cultivos
    total_pixeles_test = mascara_test_real.sum()
    if total_pixeles_test > 0:
        n_correctos = mascara_correctos.sum()
        n_errores = mascara_errores.sum()
        accuracy_total = n_correctos / total_pixeles_test
        
        # Métrica específica para el agro: ¿Qué porcentaje de los cultivos reales se acertaron?
        total_cultivos = mascara_target_cultivo.sum()
        n_cultivos_ok = ((target == prediccion) & mascara_target_cultivo).sum()
        accuracy_cultivos = (n_cultivos_ok / total_cultivos) if total_cultivos > 0 else 0
        
        print(f"\n--- Métricas de la reconstrucción (True/False) ---")
        print(f"Accuracy Total en Test (incluye fondo): {accuracy_total*100:.2f}%")
        print(f"Accuracy exclusivo en áreas de Cultivos: {accuracy_cultivos*100:.2f}%")
        print(f"Píxeles Correctos: {n_correctos:,} | Píxeles con Error: {n_errores:,}")
    else:
        print("\nNo se encontraron parches de test para evaluar.")
    
    return diferencias


def visualizar_matriz_confusion(target, prediccion, dir_exp, class_names=None, ignore_class=0):
    """
    Visualiza la matriz de confusión filtrada únicamente para los cultivos de interés.
    """
    # Máscara para aislar el set de test (-1 es el vacío de entrenamiento)
    mascara_test_real = target != -1
    
    # -----------------------------------------------------------------
    # FILTRO DE CULTIVOS DE INTERÉS
    # Clases del esquema remapeado (2=Trigo ... 7=Otros cultivos)
    clases_interes = [2, 3, 4, 5, 6, 7]
    
    # Nos aseguramos de usar solo las que estén en tu lista de interés
    if class_names is not None:
        clases_validas = [c for c in clases_interes if c in class_names]
    else:
        clases_validas = clases_interes
    
    num_clases = len(clases_validas)
    # -----------------------------------------------------------------

    clase_a_idx = {clase: idx for idx, clase in enumerate(clases_validas)}
    confusion = np.zeros((num_clases, num_clases), dtype=np.int64)
    
    # Aplanado vectorial de zonas evaluadas en el set de test
    t_flat = target[mascara_test_real]
    p_flat = prediccion[mascara_test_real]
    
    # Llenar la matriz solo con las interacciones de los cultivos de interés
    for c_real in clases_validas:
        for c_pred in clases_validas:
            idx_i = clase_a_idx[c_real]
            idx_j = clase_a_idx[c_pred]
            # Cuenta cuántos píxeles de la clase_real fueron predichos como clase_pred
            confusion[idx_i, idx_j] = np.sum((t_flat == c_real) & (p_flat == c_pred))
    
    # Normalización por filas (porcentaje de acierto por cultivo)
    confusion_norm = confusion.astype(np.float32)
    row_sums = confusion_norm.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion_norm, row_sums, where=row_sums != 0)
    
    # Configuración del gráfico
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(confusion_norm, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    
    # Generar etiquetas legibles basadas en tu diccionario oficial de QGIS
    if class_names is not None:
        labels = [class_names.get(c, f"Clase {c}") for c in clases_validas]
    else:
        labels = [f"Clase {c}" for c in clases_validas]
    
    ax.set_xticks(np.arange(num_clases))
    ax.set_yticks(np.arange(num_clases))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Clase Predicha (Modelo)', fontsize=12, labelpad=10)
    ax.set_ylabel('Clase Real (Target)', fontsize=12, labelpad=10)
    ax.set_title('Matriz de Confusión de Cultivos (%)\nSet de Test - Córdoba', fontsize=13, pad=20)
    
    # Escribir los valores numéricos y porcentajes dentro de las celdas
    for i in range(num_clases):
        for j in range(num_clases):
            ax.text(j, i, f"{confusion[i, j]:,}\n({confusion_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", 
                    color="black" if confusion_norm[i, j] < 0.5 else "white",
                    fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Proporción de aciertos')
    plt.tight_layout()
    os.makedirs(dir_exp, exist_ok=True)
    plt.savefig(f"{dir_exp}/fig/matriz_confusion.png", dpi=300, bbox_inches='tight')
    print(f"Matriz de confusión (filtrada) guardada en {dir_exp}/fig/matriz_confusion.png")
    plt.close()
    

def inference_y_visualizar(cnf, model, dataset, test_indices, posiciones_parches, 
                           class_names=None, ignore_class=0):
    """Ejecuta la inferencia aislando perfectamente el espacio muestral del set de test."""
    device = cnf.device
    model.eval()
    
    alto = cnf.alto
    ancho = cnf.ancho
    size = cnf.size_parche
    
    # CORRECCIÓN: Inicializar con -1 evita que las áreas que no son de test se confundan con la clase 0
    target_test = np.full((alto, ancho), fill_value=-1, dtype=np.int64)
    prediccion = np.full((alto, ancho), fill_value=-1, dtype=np.int64)
    
    print(f"Prediciendo {len(test_indices)} parches de test...")
    
    with torch.no_grad():
        for idx in test_indices:
            x_pos, y_pos = posiciones_parches[idx]
            
            parche_x, parche_y = dataset[idx]
            parche_x = parche_x.unsqueeze(0).to(device)
            
            output = model(parche_x)
            _, pred = torch.max(output, dim=1)
            pred = pred.squeeze(0).cpu().numpy()
            
            target_test[y_pos:y_pos+size, x_pos:x_pos+size] = parche_y.numpy()
            prediccion[y_pos:y_pos+size, x_pos:x_pos+size] = pred
    
    # Ejecutar la función de comparación (esta debe manejar target != -1 como la zona válida de test)
    diferencias = visualizar_comparacion(target_test, prediccion, cnf.dir_exp, ignore_class)
    visualizar_matriz_confusion(target_test, prediccion, cnf.dir_exp, class_names, ignore_class)
    
    return target_test, prediccion

def cargar_modelo_y_predecir(cnf):
    """Carga el modelo entrenado y ejecuta la inferencia usando la leyenda oficial de QGIS."""
    model_path = f"{cnf.dir_exp}/best_model.pth"
    test_indices_path = f"{cnf.dir_exp}/test_indices.npy"
    dat_path = os.path.dirname(cnf.file_dataset)
    posiciones_path = f"{dat_path}/posiciones_parches.npy"
    
    test_indices = np.load(test_indices_path)
    posiciones_parches = np.load(posiciones_path)
    
    dataset = CordobaDataset(cnf.file_dataset, normalizar=cnf.normalizar, label_map=cnf.label_remap)
    muestra_x, _ = dataset[0]
    in_channels = muestra_x.shape[0]

    print(f"Cargando modelo desde: {model_path}")
    model = SimpleUNet(in_channels, cnf.num_classes).to(cnf.device)

    model.load_state_dict(torch.load(model_path, map_location=cnf.device))
    model.eval()
    print("Modelo cargado correctamente")

    class_names = cnf.class_names
    
    target_test, prediccion = inference_y_visualizar(
        cnf, model, dataset, test_indices, posiciones_parches, 
        class_names=class_names, ignore_class=0
    )
    
    return target_test, prediccion



def generar_mapa_clasificacion(cnf, dir_composites, ruta_mascara, ruta_salida, tile=None):
    """
    Inferencia sobre un tile completo leyendo composites mensuales con rasterio.
    Escribe el resultado como un GeoTIFF de 1 banda (clase por píxel).
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio no está instalado. Instalalo con: pip install rasterio")

    model_path = f"{cnf.dir_exp}/best_model.pth"
    size = cnf.size_parche

    archivos_mensuales = sorted([
        f for f in os.listdir(dir_composites)
        if (tile is None or f.startswith(tile)) and f.endswith('.tif')
    ])
    if not archivos_mensuales:
        raise FileNotFoundError(f"No se encontraron composites en {dir_composites}" +
                                (f" para tile {tile}" if tile else ""))

    in_channels = len(archivos_mensuales) * 4
    print(f"Autodetectados {in_channels} canales ({len(archivos_mensuales)} meses).")

    model = SimpleUNet(in_channels, cnf.num_classes).to(cnf.device)
    model.load_state_dict(torch.load(model_path, map_location=cnf.device))
    model.eval()
    print(f"Modelo cargado desde: {model_path}")

    with rasterio.open(ruta_mascara) as src_ref:
        meta = src_ref.meta.copy()
        alto, ancho = src_ref.height, src_ref.width
    meta.update(dtype=rasterio.uint8, count=1, nodata=0)

    print(f"Dimensiones del mapa a predecir: {ancho} x {alto} píxeles.")
    print("Prediciendo por bloques...")

    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)

    with rasterio.open(ruta_salida, 'w', **meta) as dst:
        with torch.no_grad():
            for y in range(0, alto, size):
                for x in range(0, ancho, size):
                    w = min(size, ancho - x)
                    h = min(size, alto - y)
                    ventana = Window(x, y, w, h)

                    parches = []
                    for archivo_mes in archivos_mensuales:
                        with rasterio.open(os.path.join(dir_composites, archivo_mes)) as src_mes:
                            parches.append(src_mes.read(window=ventana))

                    parche_x = np.concatenate(parches, axis=0).astype(np.float32) / 10000.0
                    tensor_x = torch.from_numpy(parche_x).unsqueeze(0).to(cnf.device)

                    salida = model(tensor_x)
                    pred = torch.argmax(salida, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    dst.write(pred, 1, window=ventana)

    print(f"Mapa de clasificación guardado en: {ruta_salida}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inferencia del modelo de cultivos")
    parser.add_argument("--modo", choices=["test", "mapa"], default="test",
                        help="'test': evalúa sobre el set de test del dataset; "
                             "'mapa': genera un GeoTIFF completo sobre el tile")
    parser.add_argument("--composites", default=None, help="Directorio de composites mensuales (modo mapa)")
    parser.add_argument("--mascara", default=None, help="Ruta al TIF de referencia espacial (modo mapa)")
    parser.add_argument("--salida", default=None, help="Ruta de salida del GeoTIFF (modo mapa)")
    parser.add_argument("--tile", default=None, help="Prefijo del tile a filtrar en composites (modo mapa)")
    args = parser.parse_args()

    cnf = Cnf()

    if args.modo == "test":
        target_test, prediccion = cargar_modelo_y_predecir(cnf)
        print("\n¡Inference completada!")
        print(f"Resultados guardados en {cnf.dir_exp}")

    elif args.modo == "mapa":
        if not args.composites or not args.mascara or not args.salida:
            parser.error("--modo mapa requiere --composites, --mascara y --salida")
        generar_mapa_clasificacion(cnf, args.composites, args.mascara, args.salida, tile=args.tile)
        print("\n¡Mapa de clasificación generado!")
