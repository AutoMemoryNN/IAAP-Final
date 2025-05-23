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

    def show_bbox(
        self,
        img: np.ndarray,
        labels: List[Tuple],
        axis=None,
        show_class_names: bool = True,
        thickness: int = 2,
    ):
        """
        Dibuja cajas delimitadoras en una imagen

        Args:
            img: imagen numpy array (RGB)
            labels: lista de tuplas (class_id, x_center, y_center, width, height)
            axis: matplotlib axis para mostrar la imagen
            show_class_names: si mostrar nombres de las clases
            thickness: grosor de las líneas de las cajas
        """
        img_display = img.copy()
        img_height, img_width = img.shape[:2]

        for label in labels:
            class_id, x_center, y_center, width, height = label

            # Convertir coordenadas YOLO a bbox
            x1, y1, x2, y2 = self._yolo_to_bbox(
                (x_center, y_center, width, height), img_width, img_height
            )

            # Obtener color y nombre de clase
            color = self.colors[class_id % len(self.colors)]
            class_name = (
                self.classes[class_id]
                if class_id < len(self.classes)
                else f"class_{class_id}"
            )

            # Dibujar rectángulo
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, thickness)

            # Añadir texto con el nombre de la clase
            if show_class_names:
                # Calcular posición del texto
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 25

                # Dibujar fondo para el texto
                (text_width, text_height), _ = cv2.getTextSize(
                    class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    img_display,
                    (x1, text_y - text_height - 5),
                    (x1 + text_width, text_y + 5),
                    color,
                    -1,
                )

                # Dibujar texto
                cv2.putText(
                    img_display,
                    class_name,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        if axis is not None:
            axis.imshow(img_display)
            axis.axis("off")
        else:
            plt.figure(figsize=(12, 8))
            plt.imshow(img_display)
            plt.axis("off")
            plt.show()

        return img_display

    def visualize_random_samples(
        self,
        num_samples: int = 8,
        figsize: Tuple[int, int] = (15, 20),
        save_path: Optional[str] = None,
    ):
        """
        Visualiza muestras aleatorias del dataset

        Args:
            num_samples: número de muestras a visualizar
            figsize: tamaño de la figura
            save_path: ruta para guardar la imagen (opcional)
        """
        if len(self.image_files) == 0:
            print("No hay imágenes en el dataset")
            return

        # Seleccionar muestras aleatorias
        selected_files = random.sample(
            self.image_files, min(num_samples, len(self.image_files))
        )

        # Configurar la figura
        cols = 2
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, image_file in enumerate(selected_files):
            # Cargar imagen
            img = self._load_image(image_file)

            # Cargar etiquetas correspondientes
            label_file = self.labels_dir / f"{image_file.stem}.txt"
            labels = self._load_labels(label_file)

            # Mostrar imagen con cajas
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            self.show_bbox(img, labels, axis=ax)
            ax.set_title(f"{image_file.name}\n{len(labels)} objetos", fontsize=10)

        # Ocultar axes adicionales
        if isinstance(axes, (list, np.ndarray)) and len(selected_files) < len(axes):
            for j in range(len(selected_files), len(axes)):
                axes[j].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Imagen guardada en: {save_path}")

        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del DataManager
    dm = DataManager("raw-data")  # Ajusta la ruta según tu estructura
    
    # Mostrar información básica
    print(f"Total de imágenes encontradas: {len(dm.image_files)}")
    
    # Visualizar muestras aleatorias
    print("Visualizando muestras aleatorias del dataset...")
    dm.visualize_random_samples(num_samples=4, save_path="img/dataset_samples.png")
    
    # Ejemplo de carga y visualización de una imagen específica (si existen imágenes)
    if len(dm.image_files) > 0:
        img_path = dm.image_files[0]
        print(f"Cargando imagen de ejemplo: {img_path.name}")
        img = dm._load_image(img_path)
        
        # Cargar etiquetas correspondientes
        label_path = dm.labels_dir / f"{img_path.stem}.txt"
        labels = dm._load_labels(label_path)
        print(f"Etiquetas cargadas: {len(labels)}")
        
        # Visualizar imagen con cajas
        if labels:
            print("Mostrando imagen con cajas delimitadoras...")
            dm.show_bbox(img, labels)
        
        # Cargar etiquetas correspondientes
        label_path = dm.labels_dir / f"{img_path.stem}.txt"
        labels = dm._load_labels(label_path)
        print(f"Etiquetas cargadas: {len(labels)}")