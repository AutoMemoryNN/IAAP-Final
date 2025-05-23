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
        Initialize the DataManager

        Args:
            data_root: Path to directory containing 'images' and 'labels'
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"

        # Define the 8 classes
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

        # Colors for each class (BGR format for OpenCV)
        self.colors = [
            (0, 255, 0),  # auto - green
            (255, 0, 0),  # bus - blue
            (0, 0, 255),  # car - red
            (255, 255, 0),  # lcv - cyan
            (255, 0, 255),  # motorcycle - magenta
            (0, 255, 255),  # multiaxle - yellow
            (128, 0, 128),  # tractor - purple
            (255, 165, 0),  # truck - orange
        ]

        # Verify directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Get list of image files
        self.image_files = self._get_image_files()
        print(f"Found {len(self.image_files)} images in the dataset")

    def _get_image_files(self) -> List[Path]:
        """Get list of valid image files"""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []

        for ext in valid_extensions:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))
            image_files.extend(list(self.images_dir.glob(f"*{ext.upper()}")))

        return sorted(image_files)

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load an image"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_labels(
        self, label_path: Path
    ) -> List[Tuple[int, float, float, float, float]]:
        """
        Load YOLO labels from a file

        Returns:
            List of tuples (class_id, x_center, y_center, width, height) in normalized format
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
        Convert normalized YOLO coordinates to bounding box coordinates

        Args:
            yolo_coords: (x_center, y_center, width, height) normalized
            img_width, img_height: image dimensions

        Returns:
            (x1, y1, x2, y2) absolute coordinates
        """
        x_center, y_center, width, height = yolo_coords

        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height

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
        Draw bounding boxes on an image

        Args:
            img: numpy array image (RGB)
            labels: list of tuples (class_id, x_center, y_center, width, height)
            axis: matplotlib axis to display the image
            show_class_names: whether to show class names
            thickness: thickness of bounding box lines
        """
        img_display = img.copy()
        img_height, img_width = img.shape[:2]

        for label in labels:
            class_id, x_center, y_center, width, height = label

            x1, y1, x2, y2 = self._yolo_to_bbox(
                (x_center, y_center, width, height), img_width, img_height
            )

            color = self.colors[class_id % len(self.colors)]
            class_name = (
                self.classes[class_id]
                if class_id < len(self.classes)
                else f"class_{class_id}"
            )

            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, thickness)

            if show_class_names:
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 25

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
        Visualize random samples from the dataset

        Args:
            num_samples: number of samples to visualize
            figsize: figure size
            save_path: path to save the image (optional)
        """
        if len(self.image_files) == 0:
            print("No images in the dataset")
            return

        selected_files = random.sample(
            self.image_files, min(num_samples, len(self.image_files))
        )

        cols = 2
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, image_file in enumerate(selected_files):
            img = self._load_image(image_file)

            label_file = self.labels_dir / f"{image_file.stem}.txt"
            labels = self._load_labels(label_file)

            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            self.show_bbox(img, labels, axis=ax)
            ax.set_title(f"{image_file.name}\n{len(labels)} objects", fontsize=10)

        if isinstance(axes, (list, np.ndarray)) and len(selected_files) < len(axes):
            for j in range(len(selected_files), len(axes)):
                axes[j].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Image saved at: {save_path}")

        plt.show()

    def get_dataset_stats(
        self, show_plots: bool = True, save_path: Optional[str] = None
    ):
        """
        Analyze and display dataset statistics

        Args:
            show_plots: Whether to show statistics plots
            save_path: Base path to save plots (optional)

        Returns:
            Dictionary with dataset statistics
        """
        total_images = len(self.image_files)
        total_objects = 0
        class_counts = {cls: 0 for cls in self.classes}
        images_with_objects = 0
        objects_per_image = []
        centers_x = []
        centers_y = []

        for image_file in self.image_files:
            label_file = self.labels_dir / f"{image_file.stem}.txt"
            labels = self._load_labels(label_file)

            if labels:
                images_with_objects += 1
                total_objects += len(labels)
                objects_per_image.append(len(labels))

                for class_id, x_center, y_center, width, height in labels:
                    if class_id < len(self.classes):
                        class_counts[self.classes[class_id]] += 1

                    centers_x.append(x_center)
                    centers_y.append(y_center)
            else:
                objects_per_image.append(0)

        avg_objects = (
            total_objects / images_with_objects if images_with_objects > 0 else 0
        )
        print("=== DATASET STATISTICS ===")
        print(f"Total images: {total_images}")
        print(f"Images with objects: {images_with_objects}")
        print(f"Total objects: {total_objects}")
        print(f"Average objects per image: {avg_objects:.2f}")

        if not show_plots:
            return {
                "total_images": total_images,
                "images_with_objects": images_with_objects,
                "total_objects": total_objects,
                "avg_objects_per_image": avg_objects,
                "class_counts": class_counts,
                "objects_per_image_dist": objects_per_image,
                "centers_x": centers_x,
                "centers_y": centers_y,
            }

        fig = plt.figure(figsize=(18, 10))

        # 1. Class distribution (bars)
        ax1 = plt.subplot(1, 3, 1)
        classes_list = list(class_counts.keys())
        counts_list = list(class_counts.values())
        colors_rgb = [
            (c[2] / 255, c[1] / 255, c[0] / 255)
            for c in self.colors[: len(classes_list)]
        ]

        bars = ax1.bar(
            classes_list, counts_list, color=colors_rgb, alpha=0.8, edgecolor="black"
        )
        ax1.set_title("Class Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Vehicle Classes")
        ax1.set_ylabel("Number of Instances")
        ax1.tick_params(axis="x", rotation=45)

        for bar, count in zip(bars, counts_list):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(counts_list) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Objects per image distribution (histogram)
        ax2 = plt.subplot(1, 3, 2)
        ax2.hist(
            objects_per_image,
            bins=max(15, min(50, max(objects_per_image) + 1)),
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax2.set_title("Objects per Image Distribution", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Number of Objects per Image")
        ax2.set_ylabel("Frequency")
        ax2.axvline(
            avg_objects,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Average: {avg_objects:.1f}",
        )
        ax2.legend()

        # 3. Heatmap of box centers
        ax3 = plt.subplot(1, 3, 3)
        heatmap, xedges, yedges = np.histogram2d(
            centers_x, centers_y, bins=50, range=[[0, 1], [0, 1]]
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im = ax3.imshow(
            heatmap.T,
            extent=extent,
            origin="lower",
            cmap="hot",
            interpolation="nearest",
            aspect="auto",
        )
        ax3.set_title("Heatmap: Box Centers", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Normalized X Position")
        ax3.set_ylabel("Normalized Y Position")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_dataset_stats.png", dpi=300, bbox_inches="tight")
            print(f"Statistics saved at: {save_path}_dataset_stats.png")
        else:
            plt.savefig("img/dataset_stats.png", dpi=300, bbox_inches="tight")
            print("Statistics saved at: img/dataset_stats.png")

        plt.show()

        return {
            "total_images": total_images,
            "images_with_objects": images_with_objects,
            "total_objects": total_objects,
            "avg_objects_per_image": avg_objects,
            "class_counts": class_counts,
            "objects_per_image_dist": objects_per_image,
            "centers_x": centers_x,
            "centers_y": centers_y,
        }


if __name__ == "__main__":
    dm = DataManager("raw-data")
    print("Calculating dataset statistics...")
    dm.get_dataset_stats()
