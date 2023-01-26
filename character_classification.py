import os


import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.models as models
import cv2
import numpy as np
import helper
from pathlib import Path

BATCH_SIZE = 128



class AlphaDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform

        characters = []
        char_dirs = list(self.img_dir.iterdir())
        char_dirs = sorted(char_dirs, key=lambda x: x.name)


        for idx, char_type in enumerate(char_dirs):
            for char_path in char_type.iterdir():
                img =  cv2.imread(str(char_path))
                img = helper.resize_with_pad(img, (128, 128))
                characters.append([idx, char_type.name, img]) 

        self.samples = characters

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx, label, img = self.samples[idx]

        if self.transform:
            img = self.transform(img)
        
        return idx, label, img


class LitAlphabet(LightningModule):
    def __init__(self, data_dir, learning_rate=0.0001):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate

        num_classes = len(list(Path(data_dir).iterdir()))
        print("num_classes:", num_classes)

        # Hardcode some dataset specific attributes
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.5),
            ]
        )

        # Define PyTorch model
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        y, label, x = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        y, label, x = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


    def setup(self, stage=None):
        alpha_full = AlphaDataset(self.data_dir, transform=self.transform)
        self.alpha_train, self.alpha_val = random_split(alpha_full, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.alpha_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.alpha_val, batch_size=BATCH_SIZE)

def main():
    model_name = "circle"
    model = LitAlphabet(f"./dataset_clean/{model_name}")

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=200,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            ModelCheckpoint(save_top_k=1, filename="best")
        ],
        logger=CSVLogger(save_dir=f"logs_{model_name}/"),
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()



