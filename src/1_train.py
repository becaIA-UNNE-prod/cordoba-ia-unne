import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# Aseguramos que Python encuentre la carpeta 'utils' independientemente de dónde ejecutemos el script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import get_dataloaders
from utils.model import UNet

# --- CONFIGURACIÓN E HIPERPARÁMETROS ---
DATASET_DIR = "/mnt/yacy_1/prod/ferreyra/unet_dataset/"
MODEL_OUTPUT_DIR = Path("/mnt/yacy_1/prod/ferreyra/modelos_guardados/")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Detección de GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Dispositivo de entrenamiento: {DEVICE}")

def calculate_iou(preds, labels):
    """Calcula el Intersection over Union (IoU) para la clasificación binaria."""
    preds = torch.sigmoid(preds) > 0.5  # Convertir logits a 0 o 1
    preds = preds.int()
    labels = labels.int()
    
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    
    # Evitar división por cero
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    # tqdm genera una barra de progreso interactiva en la terminal
    pbar = tqdm(loader, desc="Entrenando", leave=False)
    for images, masks in pbar:
        images = images.to(DEVICE)
        # PyTorch espera que las máscaras binarias para BCEWithLogitsLoss sean flotantes
        masks = masks.to(DEVICE).float().unsqueeze(1) 
        
        # 1. Forward pass
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        # 2. Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 3. Métricas
        running_loss += loss.item()
        running_iou += calculate_iou(predictions, masks)
        
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(loader), running_iou / len(loader)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(loader, desc="Validando", leave=False)
    for images, masks in pbar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).float().unsqueeze(1)
        
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        running_loss += loss.item()
        running_iou += calculate_iou(predictions, masks)
        
    return running_loss / len(loader), running_iou / len(loader)

def main():
    # 1. Preparar Dataloaders
    print("Cargando datos...")
    train_loader, val_loader, _ = get_dataloaders(DATASET_DIR, batch_size=BATCH_SIZE)
    print(f"Lotes por epoch: Train={len(train_loader)} | Val={len(val_loader)}")
    
    # 2. Inicializar Modelo, Loss y Optimizador
    model = UNet(in_channels=4, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # 3. Bucle de Entrenamiento
    print("\n🚀 Iniciando entrenamiento...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_iou = validate(model, val_loader, criterion)
        
        print(f"Train -> Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"Val   -> Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")
        
        # 4. Guardar el mejor modelo (Model Checkpointing)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = MODEL_OUTPUT_DIR / "unet_sentinel_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"💾 ¡Nuevo mejor modelo guardado! (Mejora en Loss: {val_loss:.4f})")

if __name__ == "__main__":
    main()
