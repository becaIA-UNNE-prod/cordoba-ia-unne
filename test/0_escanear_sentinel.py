import os
import json
from pathlib import Path

def mapear_estructura_sentinel(base_path):
    print(f"Escaneando directorio: {base_path}...\n")
    
    estructura = {}
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"Error: No se encontró el directorio {base_path}")
        return None

    total_archivos = 0

    # 1. Iterar sobre las carpetas de los Tiles (ej. T19HGB)
    for tile_dir in base_dir.iterdir():
        if tile_dir.is_dir():
            tile_name = tile_dir.name
            estructura[tile_name] = {}
            
            # 2. Iterar sobre las carpetas de fechas (ej. 20190103)
            for date_dir in tile_dir.iterdir():
                if date_dir.is_dir():
                    date_name = date_dir.name
                    estructura[tile_name][date_name] = []
                    
                    # 3. Iterar sobre los archivos .jp2 (Bandas)
                    for file_path in date_dir.glob('*.jp2'):
                        # Extraer solo el nombre de la banda (ej. de T19HGB_20190103_B02.jp2 saca 'B02')
                        banda = file_path.stem.split('_')[-1]
                        estructura[tile_name][date_name].append(banda)
                        total_archivos += 1
                        
                    # Ordenar alfabéticamente las bandas para mayor prolijidad
                    estructura[tile_name][date_name].sort()

    return estructura, total_archivos

if __name__ == "__main__":
    DIRECTORIO_BASE = "/mnt/yacy_1/prod/ferreyra/sentinel2_cordoba/"
    
    datos, total = mapear_estructura_sentinel(DIRECTORIO_BASE)
    
    if datos:
        # Generar un resumen rápido en consola
        print("=== RESUMEN DEL DATASET ===")
        print(f"Total de Tiles encontrados: {len(datos)}")
        for tile, fechas in datos.items():
            print(f" - {tile}: {len(fechas)} fechas disponibles.")
        print(f"Total de archivos de bandas (.jp2) indexados: {total}\n")
        
        # Exportar a JSON para tener un registro estructurado
        output_file = "estructura_sentinel2.json"
        with open(output_file, 'w') as f:
            json.dump(datos, f, indent=4)
        
        print(f"Estructura completa guardada en: {output_file}")
