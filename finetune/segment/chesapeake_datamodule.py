"""
DataModule for the Chesapeake Bay dataset for segmentation tasks.

This implementation provides a structured way to handle the data loading and
preprocessing required for training and validating a segmentation model.

Dataset citation:
Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition
(CVPR 2019).

Dataset URL: https://lila.science/datasets/chesapeakelandcover
"""

import re
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class ChesapeakeDataset(Dataset):
    def __init__(self, chip_dir, label_dir, metadata, platform, target_size=(224, 224)):
        self.chip_dir = Path(chip_dir)
        self.label_dir = Path(label_dir)
        self.metadata = metadata
        self.target_size = target_size
        self.transform = self.create_transforms(
            mean=list(metadata[platform].bands.mean.values()),
            std=list(metadata[platform].bands.std.values()),
        )

        # Load chip and label file names
        # Let's do all the training data
        self.chips = sorted([chip_path.name for chip_path in self.chip_dir.glob("*.npy")])
        # self.chips = sorted([chip_path.name for chip_path in self.chip_dir.glob("*.npy")])[:1000]
        self.labels = [re.sub("_chip", "_lulc_chip", chip) for chip in self.chips]

        # Create label remapping
        self._create_label_mapping()


    # This is required because MulticlassJaccardIndex and F1Score assume consecutive label values
    # But that is not guaranteed here since not all chips have all the classes
    def _create_label_mapping(self):
        """Create mapping from original labels to consecutive integers."""
        # Load a few samples to get unique label values
        unique_labels = set()
        for label_file in self.labels[:10]:  # Check first 10 files for speed
            label_path = self.label_dir / label_file
            label = np.load(label_path)
            unique_labels.update(np.unique(label))

        # Sort unique labels to ensure consistent mapping
        sorted_labels = sorted(unique_labels)

        # Create mapping (keeping 0 as 0 if it exists)
        if 0 in sorted_labels:
            sorted_labels.remove(0)
            self.label_map = {0: 0}  # Keep 0 mapped to 0
            for i, label in enumerate(sorted_labels, start=1):
                self.label_map[label] = i
        else:
            self.label_map = {label: i for i, label in enumerate(sorted_labels)}

        # Store reverse mapping for reference
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Store number of classes
        self.num_classes = len(self.label_map)

        # print("Label mapping created:")
        # print(f"Original labels: {sorted_labels}")
        # print(f"Mapping: {self.label_map}")
        # print(f"Number of classes: {self.num_classes}")

    def create_transforms(self, mean, std):
        return v2.Compose([
            v2.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        try:
            # Load chip and label
            chip_path = self.chip_dir / self.chips[idx]
            label_path = self.label_dir / self.labels[idx]

            # Load arrays
            chip = np.load(chip_path).astype(np.float32)  # [6, 224, 224]
            label = np.load(label_path)  # Variable shape

            # Handle NaN values in chip
            chip = np.nan_to_num(chip, 0)

            # Remap labels to consecutive integers
            remapped_label = np.zeros_like(label)
            for orig_val, new_val in self.label_map.items():
                remapped_label[label == orig_val] = new_val

            # Ensure label is 3D with channel dimension
            if remapped_label.ndim == 2:
                remapped_label = np.expand_dims(remapped_label, axis=0)

            # Convert to tensors
            chip_tensor = torch.from_numpy(chip)
            label_tensor = torch.from_numpy(remapped_label).long()

            # Apply transforms to chip
            chip_tensor = self.transform(chip_tensor)

            return {
                "pixels": chip_tensor,  # [6, 224, 224]
                "label": label_tensor,  # [1, 224, 224]
                "time": torch.zeros(4, dtype=torch.float32),  # [4]
                "latlon": torch.zeros(4, dtype=torch.float32),  # [4]
            }

        except Exception as e:
            print(f"Error loading sample {idx} from {chip_path}: {str(e)}")
            raise

class ChesapeakeDataModule(L.LightningDataModule):
    """
    DataModule class for the Chesapeake Bay dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        train_label_dir (str): Directory containing training labels.
        val_chip_dir (str): Directory containing validation image chips.
        val_label_dir (str): Directory containing validation labels.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        platform (str): Platform identifier used in metadata.
    """

    def __init__(  # noqa: PLR0913
        self,
        train_chip_dir,
        train_label_dir,
        val_chip_dir,
        val_label_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
    ):
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.train_label_dir = train_label_dir
        self.val_chip_dir = val_chip_dir
        self.val_label_dir = val_label_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.platform = platform

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage (str): Stage identifier ('fit' or 'test').
        """
        if stage in {"fit", None}:
            self.trn_ds = ChesapeakeDataset(
                self.train_chip_dir,
                self.train_label_dir,
                self.metadata,
                self.platform,
            )
            self.val_ds = ChesapeakeDataset(
                self.val_chip_dir,
                self.val_label_dir,
                self.metadata,
                self.platform,
            )

    def train_dataloader(self):
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
