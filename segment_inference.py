import os
import numpy as np
import rasterio
from rasterio.transform import Affine
import torch
import torch.nn.functional as F
import fiftyone as fo
from fiftyone import ViewField as VF

from finetune.segment.chesapeake_datamodule import ChesapeakeDataModule
from finetune.segment.chesapeake_model import ChesapeakeSegmentor

# This points to the output from my fine tuning
CHESAPEAKE_CHECKPOINT_PATH = (
    "checkpoints/segment/sc-county-lulc-segment_epoch-93_val-iou-0.1420.ckpt"
)
CLAY_CHECKPOINT_PATH = "checkpoints/clay-v1.5.ckpt"

FIFTYONE_DATASET_NAME = "clay_data"

METADATA_PATH = "configs/metadata.yaml"

# Only need a dir for prediction chips
TEST_CHIP_DIR = "/home/spousty/git/clay/model/data/test/chips/"
# Only need this bc we don't want to modify the code in datamodule
TEST_LABEL_DIR = "/home/spousty/git/clay/model/data/test/labels/"

ORIGINAL_CHIP_DIR = "/home/spousty/data/remote-sensing-comparison/multi-band-image-chips"
OUTPUT_PRED_DIR = TEST_LABEL_DIR


BATCH_SIZE = 225
NUM_WORKERS = 1
PLATFORM = "hls"


def get_model(chesapeake_checkpoint_path, clay_checkpoint_path, metadata_path):
    model = ChesapeakeSegmentor.load_from_checkpoint(
        checkpoint_path=chesapeake_checkpoint_path,
        metadata_path=metadata_path,
        ckpt_path=clay_checkpoint_path,
    )
    model.eval()
    return model

def get_data(
        train_chip_dir,
        train_label_dir,
        val_chip_dir,
        val_label_dir,
        metadata_path,
        batch_size,
        num_workers,
        platform,
):
    # Handled the reduced actual paths when I call this below
    dm = ChesapeakeDataModule(
        train_chip_dir=train_chip_dir,
        train_label_dir=train_label_dir,
        val_chip_dir=train_chip_dir,
        val_label_dir=train_label_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        platform=platform,
    )
    dm.setup(stage="fit")
    val_dl = iter(dm.val_dataloader())
    batch = next(val_dl)
    metadata = dm.metadata
    chip_file_names = dm.trn_ds.chips
    return batch, metadata, chip_file_names

def run_prediction(model, batch):
    with torch.no_grad():
        outputs = model(batch)
    outputs = F.interpolate(
        outputs, size=(224, 224), mode="bilinear", align_corners=False
    )
    return outputs

def post_process(batch, outputs, metadata):
    # Convert the logits to the most likely class
    preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
    return preds

'''
All this will be is creating a png mask file from the predicition, save to disk, add the prediction label to the sample, save the sample'''
def write_png(image_array, filename):

    png_filename = filename.replace("_chip", "_pred_chip").replace(".npy", ".png")
    full_png_filename = os.path.join(OUTPUT_PRED_DIR, png_filename)

    # Create the raster file
    with rasterio.open(
            full_png_filename,
            'w',
            driver='PNG',
            height=224,
            width=224,
            count=1,  # number of bands
            dtype=np.uint8,
            transform=None,
            crs=None,
            compress=None
    ) as dst:
        dst.write(image_array,1)

    return full_png_filename


def add_to_dataset(file_names, predictions, dataset):

    if dataset.has_field("prediction"):
        dataset.delete_sample_field("prediction")

    dataset.set_values(
        "prediction",
        [fo.Segmentation() for _ in range(len(dataset))],
    )

    # iterate through the predictions = i
    for index in range(0, len(predictions)):
        file_name = file_names[index]
        image_array = predictions[index]
        full_pred_path = write_png(image_array, file_name)
        original_chip_file = os.path.join(ORIGINAL_CHIP_DIR, file_name)
        sample = dataset.select_by("chip_path", [original_chip_file]).first()
        sample["prediction.mask_path"] = full_pred_path
        sample.save()
        print("here")
    # get the filename for filenames[i]
    # save the prediction numpy array to a png file in the OUTPUT_PRED_DIR labels dir except lulc is replaced with pred
    # Make the full path to the chip for the image file = ORIGINAL_CHIP_DIR + filename
    # get the sample with that chip path
    # Make a Segmentation Label "prediction" for the sample pointing to the pred png
    # add it to the sample
    # save

if __name__ == '__main__':
    # Load Dataset
    dataset = fo.load_dataset(FIFTYONE_DATASET_NAME)

    # Load model
    model = get_model(CHESAPEAKE_CHECKPOINT_PATH, CLAY_CHECKPOINT_PATH, METADATA_PATH)

    # Get data - the dirs are all set equal because we are only doing inference
    batch, metadata, chip_file_names = get_data(
        TEST_CHIP_DIR,
        TEST_LABEL_DIR,
        TEST_CHIP_DIR,
        TEST_LABEL_DIR,
        METADATA_PATH,
        BATCH_SIZE,
        NUM_WORKERS,
        PLATFORM,
    )
    # Move batch to GPU
    batch = {k: v.to("cuda") for k, v in batch.items()}

# Run prediction
outputs = run_prediction(model, batch)

preds = post_process(batch, outputs, metadata)

# Add to FiftyOne - preds contains probability per class so pick it and add it to the sample along with the confidence
# batch["pixels"] is the stack of 6 band numpy arrays for the original image
add_to_dataset(chip_file_names, preds, dataset)

print("finished")