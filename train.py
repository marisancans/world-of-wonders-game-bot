import os


import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.models as models
import cv2
from pathlib import Path

BATCH_SIZE = 12
PATH_DATASETS = "./dataset/cells"


class AlphaDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform

        characters = []
        char_dirs = list(self.img_dir.iterdir())


        for idx, char_type in enumerate(char_dirs):
            if char_type.name == "unknown":
                continue

            for char_path in char_type.iterdir():
                img =  cv2.imread(str(char_path))
                characters.append([idx, char_type.name, img]) 

        self.samples = characters

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx, label, img = self.samples[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        
        return idx, label, img


class LitAlphabet(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, num_classes=22, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((128, 128))
            ]
        )

        # Define PyTorch model
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        y, label, x = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        y, label, x = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

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




model = LitAlphabet()
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=1000,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=CSVLogger(save_dir="logs/"),
)
trainer.fit(model)