import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union


class DataManager:
    def __init__(self, data_root: str = "raw-data"):
        """
        Inicializa el DataManager

        Args:
            data_root: Ruta al directorio que contiene 'images' y 'labels'
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"

        # Definir las 8 clases
        self.classes = [
            "auto",
            "bus",
            "car",
            "lcv",
            "motorcycle",
            "multiaxle",
            "tractor",
            "truck",
        ]

        # Colores para cada clase (BGR format para OpenCV)
        self.colors = [
            (0, 255, 0),  # auto - verde
            (255, 0, 0),  # bus - azul
            (0, 0, 255),  # car - rojo
            (255, 255, 0),  # lcv - cian
            (255, 0, 255),  # motorcycle - magenta
            (0, 255, 255),  # multiaxle - amarillo
            (128, 0, 128),  # tractor - púrpura
            (255, 165, 0),  # truck - naranja
        ]

        # Obtener lista de archivos de imágenes
        self.image_files = self._get_image_files()
        print(f"Encontradas {len(self.image_files)} imágenes en el dataset")

    def _get_image_files(self) -> List[Path]:
        """Obtiene lista de archivos de imágenes válidos"""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []

        for ext in valid_extensions:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))
            image_files.extend(list(self.images_dir.glob(f"*{ext.upper()}")))

        return sorted(image_files)


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del DataManager
    dm = DataManager("raw-data")  # Ajusta la ruta según tu estructura
    
    # Mostrar información básica
    print(f"Total de imágenes encontradas: {len(dm.image_files)}")