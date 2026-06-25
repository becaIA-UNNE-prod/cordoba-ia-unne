import torch
import numpy as np
import os, shutil
from utils import utils

class Cnf:
    """ Todos los parametros de configuracion van aqui """
    # Datos
    base_dir = "/mnt/yacy_1/prod/ferreyra/dataset"
    file_dataset = f"{base_dir}/train/dataset_20JLL.npz"
    dir_exp = f"{base_dir}/exp6"
        
    # Remapeo de etiquetas originales (0-27) al esquema simplificado:
    #   0 = NoData
    #   1 = No cultivos  (clases originales 1-14)
    #   2 = Trigo        (15)
    #   3 = Maíz         (16)
    #   4 = Soja         (17)
    #   5 = Maní         (18)
    #   6 = Sorgo        (19)
    #   7 = Otros cultivos (20-27)
    label_remap = np.array(
        [0] + [1]*14 + [2, 3, 4, 5, 6] + [7]*8,
        dtype=np.int64
    )
    class_names = {
        0: "NoData",
        1: "No cultivos",
        2: "Trigo",
        3: "Maíz",
        4: "Soja",
        5: "Maní",
        6: "Sorgo",
        7: "Otros cultivos",
    }

    # Entrenamiento
    batch_size = 16
    epochs = 100
    learning_rate = 5e-4
    num_classes = 8
    patience = 50  # early stopping patience
        
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Normalización
    normalizar = True
        
    # Semilla para reproducibilidad
    seed = 42
    
   # Dimensiones de la imagen (para inference)
    alto = 10980
    ancho = 10980
    size_parche = 256
    step_parche = 256
    
os.makedirs(Cnf.dir_exp, exist_ok=True)
dest = os.path.join(Cnf.dir_exp, os.path.basename(__file__))
if not os.path.exists(dest): shutil.copy(__file__, Cnf.dir_exp)
#shutil.copy(os.path.abspath(__file__), dir_exp)
#utils.log_experiment_info(dir_exp,__file__)
