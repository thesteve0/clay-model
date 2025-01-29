"""
LightningModule for training and validating a segmentation model using the
Segmentor class.
"""

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import F1Score, MulticlassJaccardIndex

from finetune.segment.factory import Segmentor


class ChesapeakeSegmentor(L.LightningModule):
    """
    LightningModule for segmentation tasks, utilizing Clay Segmentor.

    Attributes:
        model (nn.Module): Clay Segmentor model.
        loss_fn (nn.Module): The loss function.
        iou (Metric): Intersection over Union metric.
        f1 (Metric): F1 Score metric.
        lr (float): Learning rate.
    """

    def __init__(self, num_classes, ckpt_path, lr, wd, b1, b2):
        super().__init__()
        self.save_hyperparameters()

        self.model = Segmentor(
            num_classes=num_classes,
            ckpt_path=ckpt_path,
        )

        # Initialize loss function with ignore_index=0
        self.loss_fn = smp.losses.FocalLoss(mode="multiclass", ignore_index=0)

        # Initialize metrics with ignore_index=0
        metric_kwargs = {
            'num_classes': num_classes,
            'average': 'macro',
            'ignore_index': 0,  # Ignore zero values
            'validate_args': False
        }

        self.iou = MulticlassJaccardIndex(**metric_kwargs)
        self.f1 = F1Score(task="multiclass", **metric_kwargs)

        # Store for use in forward/training
        self.num_classes = num_classes

    def forward(self, datacube):
        """
        Forward pass through the segmentation model.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """

        waves = torch.tensor([0.493, 0.56, 0.665, 0.842, 1.61, 2.19])  # HLS wavelengths
        gsd = torch.tensor(30.0)  # HLS GSD

        # Forward pass through the network
        return self.model(
            {
                "pixels": datacube["pixels"],
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            },
        )

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=1,
            eta_min=self.hparams.lr * 100,
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training and validation.
        """
        # Get labels - should be [B, 1, H, W]
        labels = batch["label"]

        # Forward pass - should get [B, num_classes, H, W]
        outputs = self(batch)

        # Resize outputs if needed
        if outputs.shape[-2:] != (224, 224):
            outputs = F.interpolate(
                outputs,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        # Print shapes and ranges for debugging
        # print(f"\nBatch {batch_idx} Debug Info:")
        # print(f"outputs shape: {outputs.shape}")  # Should be [B, 110, 224, 224]
        # print(f"labels shape: {labels.shape}")  # Should be [B, 1, 224, 224]

        # For loss and metrics, we need [B, H, W] for labels
        labels_2d = labels.squeeze(1)

        # Calculate loss
        loss = self.loss_fn(outputs, labels_2d)

        # Get predicted classes
        preds = torch.argmax(outputs, dim=1)  # [B, H, W]

        # Debug value ranges
        # print(f"Labels range: {labels_2d.min().item()} to {labels_2d.max().item()}")
        # print(f"Predictions range: {preds.min().item()} to {preds.max().item()}")

        # Calculate metrics
        try:
            iou = self.iou(preds, labels_2d)
            f1 = self.f1(preds, labels_2d)
        except Exception as e:
            print(f"Error in metric calculation: {str(e)}")
            print(f"Unique label values: {torch.unique(labels_2d).tolist()}")
            print(f"Unique prediction values: {torch.unique(preds).tolist()}")
            raise

        # Log metrics
        self.log(
            f"{phase}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/iou",
            iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss


    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")
