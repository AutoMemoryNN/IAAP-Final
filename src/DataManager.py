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

        # Verificar que los directorios existen
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Directorio de imágenes no encontrado: {self.images_dir}"
            )
        if not self.labels_dir.exists():
            raise FileNotFoundError(
                f"Directorio de etiquetas no encontrado: {self.labels_dir}"
            )

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

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Carga una imagen"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_labels(
        self, label_path: Path
    ) -> List[Tuple[int, float, float, float, float]]:
        """
        Carga las etiquetas YOLO desde un archivo

        Returns:
            Lista de tuplas (class_id, x_center, y_center, width, height) en formato normalizado
        """
        if not label_path.exists():
            return []

        labels = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append((class_id, x_center, y_center, width, height))

        return labels

    def _yolo_to_bbox(
        self,
        yolo_coords: Tuple[float, float, float, float],
        img_width: int,
        img_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Convierte coordenadas YOLO normalizadas a coordenadas de caja delimitadora

        Args:
            yolo_coords: (x_center, y_center, width, height) normalizadas
            img_width, img_height: dimensiones de la imagen

        Returns:
            (x1, y1, x2, y2) coordenadas absolutas
        """
        x_center, y_center, width, height = yolo_coords

        # Convertir a coordenadas absolutas
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height

        # Calcular esquinas
        x1 = int(x_center_abs - width_abs / 2)
        y1 = int(y_center_abs - height_abs / 2)
        x2 = int(x_center_abs + width_abs / 2)
        y2 = int(y_center_abs + height_abs / 2)

        return x1, y1, x2, y2


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del DataManager
    dm = DataManager("raw-data")  # Ajusta la ruta según tu estructura
    
    # Mostrar información básica
    print(f"Total de imágenes encontradas: {len(dm.image_files)}")
    
    # Ejemplo de carga de una imagen (si existen imágenes)
    if len(dm.image_files) > 0:
        img_path = dm.image_files[0]
        print(f"Cargando imagen de ejemplo: {img_path.name}")
        img = dm._load_image(img_path)
        print(f"Dimensiones de la imagen: {img.shape}")
        
        # Cargar etiquetas correspondientes
        label_path = dm.labels_dir / f"{img_path.stem}.txt"
        labels = dm._load_labels(label_path)
        print(f"Etiquetas cargadas: {len(labels)}")