import os
import torch


import numpy as np
import cv2

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import config
import torchvision
import helper

def preprocess_mask(mask):
    r, g, b = cv2.split(mask)
    mask = r + g + b
    mask = np.clip(mask, 0, 1.0)

    return mask

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = root
        self.transform = transform

        self.images_directory = self.root / "rgb"
        self.masks_directory = self.root / "seg"

        self.filenames = [x.name for x in self.images_directory.iterdir()]


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = self.images_directory / filename
        mask_path = self.masks_directory / filename

        image = cv2.imread(str(image_path.absolute())).astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        trimap = cv2.imread(str(mask_path.absolute())).astype(np.float32) / 255.0
        trimap = cv2.cvtColor(trimap, cv2.COLOR_BGR2RGB)
        mask = preprocess_mask(trimap)

        # resize images
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        trimap = cv2.resize(trimap, (256, 256))

        # convert to other format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)
        trimap = np.expand_dims(trimap, 0)

        return image, mask, trimap


class GridModel(pl.LightningModule):

    def __init__(self, arch="FPN", encoder_name="resnet34", in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image, mask, trimap = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0


        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        if config.DEBUG and stage == "valid":
            bs = image.shape[0]
            img_gray = torchvision.transforms.functional.rgb_to_grayscale(image)
            grid = torchvision.utils.make_grid(torch.cat([img_gray, logits_mask, prob_mask, pred_mask]), nrow=bs)
            grid = grid.permute(1, 2, 0).cpu().detach().numpy()
            helper.show("grid", grid, 1)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


def main():
    root = Path("./dataset_clean/grid")

    dataset = SimpleDataset(root)
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    model_name = "grid"

    model = GridModel()
        

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10,
        # callbacks=[
        #     TQDMProgressBar(refresh_rate=20),
        #     ModelCheckpoint(monitor='valid_dataset_iou', save_top_k=1, filename="best")
        # ],
        # logger=CSVLogger(save_dir=f"logs_{model_name}/"),
        default_root_dir="./logs_grid"
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
    )


    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    return model


    image, mask, trimap = next(iter(valid_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(image)
    pr_masks = logits.sigmoid()

    import matplotlib.pyplot as plt

    for image, gt_mask, pr_mask in zip(image, mask, pr_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    main()