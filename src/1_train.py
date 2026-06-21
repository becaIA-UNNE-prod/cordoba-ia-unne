import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.append(os.path.abspath("."))

# Asumimos que estos módulos ya están creados en tu proyecto
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
    SPLIT_INDEX = "/mnt/yacy_1/prod/ferreyra/dataset/split_index.json"
    BATCH_SIZE = 4 # Vigila el uso de VRAM. Si te quedas sin memoria (OOM), bájalo a 2.
    EPOCHS = 5
    LEARNING_RATE = 1e-4

    # Asegúrate de que NUM_CLASSES sea mayor al valor máximo que tienes en tus etiquetas Y
    NUM_CLASSES = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # 1. Cargar splits desde el índice generado por 04_split_dataset.py
    train_dataset = CordobaDataset(split_index=SPLIT_INDEX, split="train")
    val_dataset   = CordobaDataset(split_index=SPLIT_INDEX, split="val")
    test_dataset  = CordobaDataset(split_index=SPLIT_INDEX, split="test")

    if len(train_dataset) == 0:
        print("Error: No hay datos de entrenamiento. Ejecutá primero 04_split_dataset.py")
        return

    print(f"Splits -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Autodetectar IN_CHANNELS leyendo el primer parche
    muestra_x, _ = train_dataset[0]
    IN_CHANNELS = muestra_x.shape[0]
    print(f"Autodetectados {IN_CHANNELS} canales de entrada (Meses x Bandas).")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleUNet(IN_CHANNELS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Bucle de Entrenamiento
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

        # 4. Fase de Validación
        val_loss_epoch, val_acc_epoch = evaluar(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"| Train Loss: {train_loss_epoch:.4f} Acc: {train_acc_epoch:.4f} "
              f"| Val Loss: {val_loss_epoch:.4f} Acc: {val_acc_epoch:.4f}")

    # 5. Fase de Testeo Final
    print("\n--- Evaluando en conjunto de TEST ---")
    test_loss, test_acc = evaluar(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # 6. Guardar pesos
    os.makedirs("./pesos", exist_ok=True)
    torch.save(model.state_dict(), "./pesos/modelo_cordoba_test.pth")
    print("Modelo guardado exitosamente.")

if __name__ == "__main__":
    entrenar()
