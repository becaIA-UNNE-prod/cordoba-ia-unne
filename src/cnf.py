import torch
import numpy as np
import os, shutil
from utils import utils

class Cnf:
    """ Todos los parametros de configuracion van aqui """
    # Datos
    base_dir = "/mnt/yacy_1/prod/ferreyra/dataset"
    dir_dataset = f"{base_dir}/train"
    dir_exp = f"{base_dir}/exp7"

    # Tiles usados para entrenar. Cada tile debe tener sus parches ya
    # extraidos en dir_dataset (ver src/03_extraer_parches.py) como:
    #   X_<tile>.npy, Y_<tile>.npy, meta_<tile>.npz, posiciones_<tile>.npy
    # Para entrenar con mas de un tile, simplemente agregarlo a esta lista
    # una vez que su extraccion haya terminado. Todos los tiles deben cubrir
    # los mismos meses (mismo numero de canales), si no el entrenamiento
    # falla con un error explicito al construir el dataset.
    tiles = ["T20JLL"]

    @staticmethod
    def paths_tile(tile_id, dir_dataset=None):
        """ Rutas de los archivos de un tile ya extraido. Usado tanto por
        03_extraer_parches.py (al escribir) como por utils/dataset.py (al leer),
        para que la convencion de nombres viva en un solo lugar. """
        d = dir_dataset if dir_dataset is not None else Cnf.dir_dataset
        return {
            "x": os.path.join(d, f"X_{tile_id}.npy"),
            "y": os.path.join(d, f"Y_{tile_id}.npy"),
            "meta": os.path.join(d, f"meta_{tile_id}.npz"),
            "posiciones": os.path.join(d, f"posiciones_{tile_id}.npy"),
        }

    # Remapeo de etiquetas originales (0-27) a esquema de 9 clases:
    #   0 = NoData
    #   1 = Natural          (1-10: monte, arbustales, pastizal, suelo desnudo, rocas, salina, agua, anegables, cursos de agua, ...)
    #   2 = Urbano/Infraest. (11-14: urbano consolidado, en consolidación, sin consolidar, vial)
    #   3 = Trigo            (15)
    #   4 = Maíz             (16)
    #   5 = Soja             (17)
    #   6 = Maní             (18)
    #   7 = Sorgo            (19)
    #   8 = Otros cultivos   (20-27: dobles, pasturas, plantaciones, frutales)
    label_remap = np.array(
        [0] + [1]*10 + [2]*4 + [3, 4, 5, 6, 7] + [8]*8,
        dtype=np.int64
    )
    class_names = {
        0: "NoData",
        1: "Natural",
        2: "Urbano/Infraestructura",
        3: "Trigo",
        4: "Maíz",
        5: "Soja",
        6: "Maní",
        7: "Sorgo",
        8: "Otros cultivos",
    }

    # Entrenamiento
    batch_size = 16
    epochs = 100
    learning_rate = 5e-4
    num_classes = 9
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
