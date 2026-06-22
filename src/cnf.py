import torch
import os, shutil
from utils import utils

class Cnf:
    """ Todos los parametros de configuracion van aqui """
    # Datos
    file_dataset = "../dat/train/dataset_20JLL.npz"
    dir_exp = "../dat/exp6"
        
    # Entrenamiento
    batch_size = 16
    epochs = 100
    learning_rate = 5e-4
    num_classes = 50
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
