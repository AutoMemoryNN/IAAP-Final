import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt


class DatasetManager:
    def __init__(
        self,
        raw_data_path: str = "raw-data",
        output_data_path: str = "data",
        class_mapping: Optional[Dict[str, str]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.output_data_path = Path(output_data_path)

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.original_classes = self._load_classes()
        self.class_mapping = class_mapping or {}
        self.final_classes = self._get_final_classes()

        self.output_data_path.mkdir(parents=True, exist_ok=True)

        print(
            f"DatasetManager initialized with {len(self.original_classes)} original classes and {len(self.final_classes)} final classes"
        )

    def _load_classes(self) -> List[str]:
        return [
            "auto",
            "bus",
            "car",
            "lcv",
            "motorcycle",
            "multiaxle",
            "tractor",
            "truck",
        ]

    def _get_final_classes(self) -> List[str]:
        if not self.class_mapping:
            return self.original_classes.copy()

        mapped_classes = set()
        for original_class in self.original_classes:
            mapped_class = self.class_mapping.get(original_class, original_class)
            mapped_classes.add(mapped_class)

        return sorted(list(mapped_classes))

    def set_class_mapping(self, class_mapping: Dict[str, str]):
        self.class_mapping = class_mapping
        self.final_classes = self._get_final_classes()
        print(f"Class mapping updated. Final classes: {self.final_classes}")

    def _map_class_id(self, original_class_id: int) -> Optional[int]:
        if original_class_id >= len(self.original_classes):
            return None

        original_class_name = self.original_classes[original_class_id]
        mapped_class_name = self.class_mapping.get(
            original_class_name, original_class_name
        )

        try:
            return self.final_classes.index(mapped_class_name)
        except ValueError:
            return None

    def _load_and_process_sample(
        self, image_path: Path
    ) -> Optional[Tuple[np.ndarray, List[List[float]]]]:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load corresponding labels
            label_path = self.raw_data_path / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                return img, []

            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            original_class_id = int(parts[0])
                            mapped_class_id = self._map_class_id(original_class_id)

                            if mapped_class_id is not None:
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])

                                labels.append(
                                    [mapped_class_id, x_center, y_center, width, height]
                                )

            return img, labels

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def split_dataset(self, random_seed: int = 42) -> Dict[str, List[Path]]:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        all_images = []

        images_dir = self.raw_data_path / "images"
        for ext in image_extensions:
            all_images.extend(list(images_dir.glob(f"*{ext}")))
            all_images.extend(list(images_dir.glob(f"*{ext.upper()}")))

        all_images = sorted(all_images)
        if len(all_images) == 0:
            raise ValueError("No images found in directory")

        random.seed(random_seed)
        np.random.seed(random_seed)

        # First split: train vs (val + test)
        train_images, temp_images = train_test_split(
            all_images,
            train_size=self.train_ratio,
            random_state=random_seed,
            shuffle=True,
        )

        # Second split: val vs test
        if self.test_ratio > 0:
            val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_images, test_images = train_test_split(
                temp_images, train_size=val_size, random_state=random_seed, shuffle=True
            )
        else:
            val_images = temp_images
            test_images = []

        splits = {"train": train_images, "val": val_images, "test": test_images}

        for split_name, images in splits.items():
            print(f"  {split_name}: {len(images)} images")

        return splits

    def process_and_save_dataset(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        random_seed: int = 42,
        save_metadata: bool = True,
    ) -> Dict[str, any]:
        print("Starting dataset processing...")

        splits = self.split_dataset(random_seed)

        stats = {
            "processed_samples": 0,
            "failed_samples": 0,
            "splits": {},
            "class_distribution": defaultdict(int),
            "target_size": target_size,
        }

        for split_name, image_paths in splits.items():
            if len(image_paths) == 0:
                continue

            print(f"\nProcessing {split_name}...")
            split_stats = self._process_split(image_paths, split_name, target_size)
            stats["splits"][split_name] = split_stats
            stats["processed_samples"] += split_stats["processed"]
            stats["failed_samples"] += split_stats["failed"]

            for class_name, count in split_stats["class_distribution"].items():
                stats["class_distribution"][class_name] += count

        if save_metadata:
            self._save_metadata(stats)

        print(
            f"\nProcessing completed: {stats['processed_samples']} samples processed, {stats['failed_samples']} failed"
        )

        return stats

    def _process_split(
        self,
        image_paths: List[Path],
        split_name: str,
        target_size: Optional[Tuple[int, int]],
    ) -> Dict[str, any]:
        processed = 0
        failed = 0
        class_distribution = defaultdict(int)

        all_images = []
        all_labels = []
        sample_names = []

        for i, image_path in enumerate(image_paths):
            result = self._load_and_process_sample(image_path)

            if result is None:
                failed += 1
                continue

            img, labels = result

            if target_size:
                img = cv2.resize(img, target_size)

            if len(labels) > 0:
                for label in labels:
                    class_id = int(label[0])
                    if class_id < len(self.final_classes):
                        class_name = self.final_classes[class_id]
                        class_distribution[class_name] += 1

            all_images.append(img)
            all_labels.append(labels)
            sample_names.append(image_path.stem)
            processed += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} samples...")

        if processed > 0:
            images_array = np.array(all_images)
            labels_array = np.array(all_labels, dtype=object)
            names_array = np.array(sample_names)

            np.save(self.output_data_path / f"{split_name}_images.npy", images_array)
            np.save(self.output_data_path / f"{split_name}_labels.npy", labels_array)
            np.save(self.output_data_path / f"{split_name}_names.npy", names_array)

        return {
            "processed": processed,
            "failed": failed,
            "class_distribution": dict(class_distribution),
            "shape": images_array.shape if processed > 0 else None,
        }

    def _save_metadata(self, stats: Dict[str, any]):
        metadata = {
            "original_classes": self.original_classes,
            "final_classes": self.final_classes,
            "class_mapping": self.class_mapping,
            "split_ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "processing_stats": stats,
        }

        metadata_path = self.output_data_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    def load_processed_split(
        self, split: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        images_path = self.output_data_path / f"{split}_images.npy"
        labels_path = self.output_data_path / f"{split}_labels.npy"
        names_path = self.output_data_path / f"{split}_names.npy"

        if not all([images_path.exists(), labels_path.exists(), names_path.exists()]):
            return None

        try:
            images = np.load(images_path)
            labels = np.load(labels_path, allow_pickle=True)
            names = np.load(names_path)
            return images, labels, names
        except Exception as e:
            print(f"Error loading split {split}: {e}")
            return None


if __name__ == "__main__":
    dm = DatasetManager()
    stats = dm.process_and_save_dataset(target_size=(224, 224))

    images, labels, names = dm.load_processed_split("train")
    if images is not None:
        print(f"Train set loaded: {images.shape}")
