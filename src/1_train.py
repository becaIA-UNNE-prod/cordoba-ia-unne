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
from utils import utils
from cnf import Cnf # configuracion
        
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

def entrenar(cnf):
    """
    Función principal de entrenamiento
    """
    # Fijar semilla para reproducibilidad
    torch.manual_seed(cnf.seed)
    np.random.seed(cnf.seed)
    
    print(f"Dispositivo: {cnf.device}")

    # 1. Cargar dataset completo
    dataset_completo = CordobaDataset(cnf.file_dataset, normalizar=cnf.normalizar, label_map=cnf.label_remap)
    total_size = len(dataset_completo)

    muestra_x, _ = dataset_completo[0]
    in_channels = muestra_x.shape[0]

    print(f"Autodetectados {in_channels} canales de entrada (Meses x Bandas).")
    print(f"Imagenes de {muestra_x.shape[1]} x {muestra_x.shape[2]}")
    
    # 2. Dividir en Train (70%), Val (15%), Test (15%)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Crear lista de índices y hacer split
    indices = list(range(total_size))
    train_indices, val_indices, test_indices = random_split(
        indices, [train_size, val_size, test_size]
    )
    
    # Guardar índices de test para inference
    os.makedirs(cnf.dir_exp, exist_ok=True)
    np.save(f"{cnf.dir_exp}/test_indices.npy", list(test_indices))
    print(f"Índices de test guardados en {cnf.dir_exp}/test_indices.npy")

    # Crear datasets usando Subset
    train_dataset = torch.utils.data.Subset(dataset_completo, train_indices)
    val_dataset = torch.utils.data.Subset(dataset_completo, val_indices)
    test_dataset = torch.utils.data.Subset(dataset_completo, test_indices)

    print(f"Splits -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cnf.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cnf.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cnf.batch_size, shuffle=False)

    # 3. Crear modelo
    model = SimpleUNet(in_channels, cnf.num_classes).to(cnf.device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=cnf.learning_rate)

    # Inicializar early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Inicializar listas para guardar métricas
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # 4. Bucle de Entrenamiento
    for epoch in range(cnf.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch_idx, (datos_x, etiquetas_y) in enumerate(train_loader):
            datos_x = datos_x.to(cnf.device)
            etiquetas_y = etiquetas_y.to(cnf.device)

            optimizer.zero_grad()
            predicciones = model(datos_x)

            loss = criterion(predicciones, etiquetas_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calcular_accuracy(predicciones, etiquetas_y)

        train_loss_epoch = train_loss / len(train_loader)
        train_acc_epoch = train_acc / len(train_loader)

        train_losses.append(train_loss_epoch)
        train_accs.append(train_acc_epoch)

        # Fase de Validación
        val_loss_epoch, val_acc_epoch = evaluar(model, val_loader, criterion, cnf.device)

        val_losses.append(val_loss_epoch)
        val_accs.append(val_acc_epoch)

        print(f"Epoch [{epoch+1}/{cnf.epochs}] "
              f"| Train Loss: {train_loss_epoch:.4f} Acc: {train_acc_epoch:.4f} "
              f"| Val Loss: {val_loss_epoch:.4f} Acc: {val_acc_epoch:.4f}")

        print('Validacion:',val_loss_epoch,best_val_loss)
        
        # Guardar el mejor modelo
        if val_loss_epoch < best_val_loss:
            os.makedirs(cnf.dir_exp, exist_ok=True)
            torch.save(model.state_dict(), f"{cnf.dir_exp}/best_model.pth")
            print(f"Mejor modelo guardado en {cnf.dir_exp}/best_model.pth")
            
        # Early Stopping
        should_stop, best_val_loss, patience_counter = early_stopping(
            val_loss_epoch, 
            best_val_loss, 
            patience_counter, 
            cnf.patience, 
            epoch
        )
        
        if should_stop:
            break
    
    # Cargar el mejor modelo para test
    best_model_path = f"{cnf.dir_exp}/best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Modelo cargado desde {best_model_path}")
    
    # 5. Fase de Testeo Final
    print("\n--- Evaluando en conjunto de TEST ---")
    test_loss, test_acc = evaluar(model, test_loader, criterion, cnf.device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # 6. Guardar modelo final
    os.makedirs(cnf.dir_exp, exist_ok=True)
    torch.save(model.state_dict(), f"{cnf.dir_exp}/modelo_final.pth")
    print(f"Modelo final guardado en {cnf.dir_exp}/modelo_final.pth")

    # 7. Graficar métricas
    graficar_metricas(train_losses, val_losses, train_accs, val_accs,
                      len(train_losses), cnf.dir_exp)
    
    # 8. Guardar métricas en archivo
    guardar_metricas(train_losses, val_losses, train_accs, val_accs, test_loss,
                     test_acc, len(train_losses), cnf.dir_exp)

    # 9. Guardar configuración
    cnf_dict = utils.obs2dict(Cnf)
    config_path = f"{cnf.dir_exp}/config.txt"
    with open(config_path, 'w') as f:
        for key, value in cnf_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"Configuracion guardada en {config_path}")
    
    
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
    os.makedirs(f"{exp_dir}/fig", exist_ok=True)
    plt.savefig(f"{exp_dir}/fig/metricas_entrenamiento.png", dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {exp_dir}/fig/metricas_entrenamiento.png")
    
def early_stopping(val_loss, best_val_loss, patience_counter, patience, epoch):

    improved = val_loss < best_val_loss
    
    if improved:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    should_stop = patience_counter >= patience
    
    return should_stop, best_val_loss, patience_counter

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
    cnf = Cnf()    
    entrenar(cnf)
    

