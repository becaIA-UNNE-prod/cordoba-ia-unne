import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append(os.path.abspath("../"))
from utils.dataset import CordobaDataset
from utils.model import SimpleUNet

def calcular_accuracy(predicciones, etiquetas, ignore_index=0):
    """Calcula el porcentaje de píxeles correctos ignorando el fondo (NoData)"""
    pred_clases = torch.argmax(predicciones, dim=1)
    mascara_validos = etiquetas != ignore_index

    correctos = (pred_clases[mascara_validos] == etiquetas[mascara_validos]).sum().item()
    total_validos = mascara_validos.sum().item()

    if total_validos == 0:
        return 0.0
    return correctos / total_validos

def evaluar(modelo, dataloader, criterion, device):
    """Función genérica para evaluar en Validación o Test"""
    modelo.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for datos_x, etiquetas_y in dataloader:
            datos_x = datos_x.to(device)
            etiquetas_y = etiquetas_y.to(device)

            salidas = modelo(datos_x)
            loss = criterion(salidas, etiquetas_y)
            acc = calcular_accuracy(salidas, etiquetas_y)

            running_loss += loss.item()
            running_acc += acc

    loss_promedio = running_loss / len(dataloader)
    acc_promedio = running_acc / len(dataloader)
    return loss_promedio, acc_promedio

def entrenar():
    DIR_DATASET = "../dat/train"
    DIR_EXP='../dat/exp2'
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # 1. Cargar dataset completo
    dataset_completo = CordobaDataset(DIR_DATASET)
    total_size = len(dataset_completo)

    muestra_x, _ = dataset_completo[0]
    IN_CHANNELS = muestra_x.shape[0]

    print(f"Autodetectados {IN_CHANNELS} canales de entrada (Meses x Bandas).")

    # 2. Dividir en Train (70%), Val (15%), Test (15%)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset_completo, [train_size, val_size, test_size]
    )

    print(f"Splits -> Train: {train_size} | Val: {val_size} | Test: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Inicializar listas para guardar métricas
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # 4. Bucle de Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch_idx, (datos_x, etiquetas_y) in enumerate(train_loader):
            datos_x = datos_x.to(device)
            etiquetas_y = etiquetas_y.to(device)

            optimizer.zero_grad()
            predicciones = model(datos_x)

            loss = criterion(predicciones, etiquetas_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calcular_accuracy(predicciones, etiquetas_y)

        train_loss_epoch = train_loss / len(train_loader)
        train_acc_epoch = train_acc / len(train_loader)

        # Guardar métricas de entrenamiento
        train_losses.append(train_loss_epoch)
        train_accs.append(train_acc_epoch)

        # 5. Fase de Validación
        val_loss_epoch, val_acc_epoch = evaluar(model, val_loader, criterion, device)

        # Guardar métricas de validación
        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"| Train Loss: {train_loss_epoch:.4f} Acc: {train_acc_epoch:.4f} "
              f"| Val Loss: {val_loss_epoch:.4f} Acc: {val_acc_epoch:.4f}")

    # 6. Fase de Testeo Final
    print("\n--- Evaluando en conjunto de TEST ---")
    test_loss, test_acc = evaluar(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # 7. Guardar modelo
    os.makedirs(DIR_EXP, exist_ok=True)
    torch.save(model.state_dict(), f"{DIR_EXP}/modelo_cordoba.pth")
    print("Modelo guardado.")

    # 8. Graficar métricas
    graficar_metricas(train_losses, val_losses, train_accs, val_accs, EPOCHS, DIR_EXP)
    
    # 9. Opcional: guardar métricas en archivo
    guardar_metricas(train_losses, val_losses, train_accs, val_accs, test_loss, test_acc, EPOCHS, DIR_EXP)

def graficar_metricas(train_losses, val_losses, train_accs, val_accs, epochs, exp_dir):
    """
    Grafica las pérdidas y precisiones de entrenamiento y validación
    """
    epochs_range = range(1, epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de pérdidas
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida (Loss)')
    ax1.set_title('Evolución de la Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de precisiones
    ax2.plot(epochs_range, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs_range, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión (Accuracy)')
    ax2.set_title('Evolución de la Precisión')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    os.makedirs(f"{exp_dir}/graficos", exist_ok=True)
    plt.savefig(f"{exp_dir}/graficos/metricas_entrenamiento.png", dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {exp_dir}/graficos/metricas_entrenamiento.png")
    

def guardar_metricas(train_losses, val_losses, train_accs, val_accs, test_loss, test_acc, epochs, exp_dir):
    """
    Guarda las métricas en un archivo .npz para análisis posterior
    """
    metricas = {
        'train_losses': np.array(train_losses),
        'val_losses': np.array(val_losses),
        'train_accs': np.array(train_accs),
        'val_accs': np.array(val_accs),
        'test_loss': test_loss,
        'test_acc': test_acc,
        'epochs': epochs,
        'best_val_acc': max(val_accs),
        'best_val_epoch': np.argmax(val_accs) + 1
    }
    
    np.savez(f"{exp_dir}/metricas_entrenamiento.npz", **metricas)
    print(f"Metricas guardadas en {exp_dir}/metricas_entrenamiento.npz")
    
    # Mostrar resumen
    print("\n--- Resumen del Entrenamiento ---")
    print(f"Mejor precisión de validación: {max(val_accs):.4f} en época {np.argmax(val_accs) + 1}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

def cargar_y_graficar_metricas(exp_dir,ruta_metricas="metricas_entrenamiento.npz"):
    """
    Función auxiliar para cargar métricas guardadas y graficarlas
    """
    data = np.load(ruta_metricas, allow_pickle=True)
    
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    train_accs = data['train_accs']
    val_accs = data['val_accs']
    epochs = int(data['epochs'])
    
    print(f"Cargadas métricas de {epochs} épocas")
    print(f"Mejor val acc: {data['best_val_acc']:.4f} en época {data['best_val_epoch']}")
    
    graficar_metricas(train_losses, val_losses, train_accs, val_accs, epochs, exp_dir)

if __name__ == "__main__":
    entrenar()
    
    # Si quieres cargar y graficar desde archivo guardado:
    # cargar_y_graficar_metricas()

