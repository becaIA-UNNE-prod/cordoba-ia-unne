import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch.utils.data import DataLoader

def visualizar_comparacion(target, prediccion, dir_exp, ignore_class=0):
    """Visualiza target, predicción y diferencias en 3 paneles"""
    mascara_valida = target != ignore_class
    
    # Calcular diferencias: 0=correcto, 1=FP, 2=FN
    diferencias = np.zeros_like(target, dtype=np.int8)
    diferencias[mascara_valida] = 1
    diferencias[mascara_valida & (target[mascara_valida] == prediccion[mascara_valida])] = 0
    diferencias[mascara_valida & (target[mascara_valida] != prediccion[mascara_valida]) & (prediccion[mascara_valida] == ignore_class)] = 2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colormap para clases
    clases_unicas = np.unique(target[target != ignore_class])
    num_clases = len(clases_unicas)
    cmap_clases = plt.cm.tab20
    norm_clases = BoundaryNorm(np.arange(-0.5, num_clases + 0.5, 1), cmap_clases.N)
    
    # Target
    axes[0].imshow(target, cmap=cmap_clases, norm=norm_clases, interpolation='none')
    axes[0].set_title('Target (Ground Truth)', fontsize=14)
    axes[0].axis('off')
    
    # Predicción
    axes[1].imshow(prediccion, cmap=cmap_clases, norm=norm_clases, interpolation='none')
    axes[1].set_title('Inferencia (Predicción)', fontsize=14)
    axes[1].axis('off')
    
    # Diferencias
    cmap_diff = ListedColormap(['#2ecc71', '#e74c3c', '#f1c40f'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm_diff = BoundaryNorm(bounds, cmap_diff.N)
    axes[2].imshow(diferencias, cmap=cmap_diff, norm=norm_diff, interpolation='none')
    axes[2].set_title('Diferencias\n(Verde=Correcto, Rojo=FP, Amarillo=FN)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(dir_exp, exist_ok=True)
    plt.savefig(f"{dir_exp}/comparacion.png", dpi=300, bbox_inches='tight')
    print(f"Comparación guardada en {dir_exp}/comparacion.png")
    plt.close()
    
    # Métricas
    total_validos = mascara_valida.sum()
    if total_validos > 0:
        correctos = (target[mascara_valida] == prediccion[mascara_valida]).sum()
        accuracy = correctos / total_validos
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        print(f"Correctos: {correctos:,} | FP: {(diferencias==1).sum():,} | FN: {(diferencias==2).sum():,}")
    
    return diferencias


def visualizar_matriz_confusion(target, prediccion, dir_exp, class_names=None, ignore_class=0):
    """Visualiza matriz de confusión"""
    mascara_valida = target != ignore_class
    clases_unicas = np.unique(target[mascara_valida])
    num_clases = len(clases_unicas)
    
    confusion = np.zeros((num_clases, num_clases), dtype=np.int64)
    for i, clase_real in enumerate(clases_unicas):
        for j, clase_pred in enumerate(clases_unicas):
            mascara = (target == clase_real) & (prediccion == clase_pred) & mascara_valida
            confusion[i, j] = mascara.sum()
    
    confusion_norm = confusion.astype(np.float32)
    row_sums = confusion_norm.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion_norm, row_sums, where=row_sums != 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_norm, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    
    if class_names is not None:
        labels = [class_names.get(c, f"Clase {c}") for c in clases_unicas]
    else:
        labels = [f"Clase {c}" for c in clases_unicas]
    
    ax.set_xticks(np.arange(num_clases))
    ax.set_yticks(np.arange(num_clases))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Clase Predicha')
    ax.set_ylabel('Clase Real')
    ax.set_title('Matriz de Confusión (%)')
    
    for i in range(num_clases):
        for j in range(num_clases):
            ax.text(j, i, f"{confusion[i, j]:,}\n({confusion_norm[i, j]*100:.1f}%)",
                   ha="center", va="center", 
                   color="black" if confusion_norm[i, j] < 0.5 else "white",
                   fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Porcentaje')
    plt.tight_layout()
    plt.savefig(f"{dir_exp}/matriz_confusion.png", dpi=300, bbox_inches='tight')
    print(f"Matriz de confusión guardada en {dir_exp}/matriz_confusion.png")
    plt.close()


def inference_y_visualizar(cnf, model, test_indices, posiciones_parches, 
                          class_names=None, ignore_class=0):
    """
    Hace inference sobre los parches de test y visualiza resultados
    
    Args:
        cnf: Configuración
        model: Modelo entrenado
        test_indices: Lista de índices de test
        posiciones_parches: Lista de (x, y) para cada parche
        class_names: Diccionario {clase: nombre} para la matriz de confusión
        ignore_class: Clase a ignorar (fondo)
    
    Returns:
        target_test: Matriz de ground truth (solo test)
        prediccion: Matriz de predicción
    """
    device = cnf.device
    model.eval()
    
    # Cargar dataset original
    dataset_completo = CordobaDataset(cnf.file_dataset, normalizar=cnf.normalizar)
    
    # Dimensiones de la imagen original
    alto, ancho = 10980, 10980
    size = 256
    
    # Crear matrices para target y predicción
    target_test = np.zeros((alto, ancho), dtype=np.int64)
    prediccion = np.zeros((alto, ancho), dtype=np.int64)
    
    print(f"Prediciendo {len(test_indices)} parches de test...")
    
    with torch.no_grad():
        for idx, (x_pos, y_pos) in zip(test_indices, posiciones_parches):
            # Obtener parche X
            parche_x, parche_y = dataset_completo[idx]
            
            # Mover a device y agregar batch dimension
            parche_x = parche_x.unsqueeze(0).to(device)  # [1, C, H, W]
            
            # Predecir
            output = model(parche_x)
            _, pred = torch.max(output, dim=1)  # [1, H, W]
            pred = pred.squeeze(0).cpu().numpy()  # [H, W]
            
            # Colocar en las matrices completas
            target_test[y_pos:y_pos+size, x_pos:x_pos+size] = parche_y.numpy()
            prediccion[y_pos:y_pos+size, x_pos:x_pos+size] = pred
    
#    # Guardar matrices
#    np.save(f"{cnf.dir_exp}/target_test.npy", target_test)
#    np.save(f"{cnf.dir_exp}/prediccion_test.npy", prediccion)
#    print(f"Target guardado en {cnf.dir_exp}/target_test.npy")
#    print(f"Predicción guardada en {cnf.dir_exp}/prediccion_test.npy")
    
    # Visualizar comparación
    diferencias = visualizar_comparacion(target_test, prediccion, cnf.dir_exp, ignore_class)
    
    # Visualizar matriz de confusión
    visualizar_matriz_confusion(target_test, prediccion, cnf.dir_exp, class_names, ignore_class)
    
    return target_test, prediccion


def cargar_modelo_y_predecir(cnf):
    """
    Carga modelo y datos, y ejecuta inference
    """
    # 1. Definir rutas
    model_path = f"{cnf.dir_exp}/best_model.pth"
    test_indices_path = f"{cnf.dir_exp}/test_indices.npy"
    posiciones_path = f"{cnf.dir_exp}/posiciones_parches.npy"
    
    # 3. Cargar índices y posiciones
    test_indices = np.load(test_indices_path)
    posiciones_parches = np.load(posiciones_path)
    
    # 4. Determinar canales de entrada
    dataset_temp = CordobaDataset(cnf.file_dataset, normalizar=cnf.normalizar)
    muestra_x, _ = dataset_temp[0]
    in_channels = muestra_x.shape[0]
    
    # 5. Cargar modelo
    print(f"Cargando modelo desde: {model_path}")
    model = SimpleUNet(in_channels, cnf.num_classes).to(cnf.device)
    model.load_state_dict(torch.load(model_path, map_location=cnf.device))
    model.eval()
    print("Modelo cargado correctamente")
    
    # 6. Ejecutar inference
    class_names = {
        1: "Maíz",
        2: "Soja",
        3: "Trigo",
        4: "Girasol",
        5: "Algodón",
        # ... agregar todas las clases
    }
    
    target_test, prediccion = inference_y_visualizar(
        cnf, model, test_indices, posiciones_parches, 
        class_names=class_names, ignore_class=0
    )
    
    return target_test, prediccion


if __name__ == "__main__":
    # 1. Configuración
    cnf = Cnf()
    cnf.file_dataset = "../dat/train/dataset_20JLL.npz"
    cnf.dir_exp = "../dat/exp4"
    cnf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Cargar modelo y hacer inference
    target_test, prediccion = cargar_modelo_y_predecir(cnf)
    
    print("\n¡Inference completada!")
    print(f"Resultados guardados en {cnf.dir_exp}")
