import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
