import torch
import os

from trainer import Trainer
from utils import save_results
from data.custom_transforms import base_transform, random_transform
from data.make_dataset import SegmentationDataModule
from models.test_cnn import TestCNN
from models.encoder_decoder import EncDec_base, EncDecStride, EncDec_dropout, DilatedConvNet
from models.unet import UNetDeconv, UNetDilated
from loss_functions.focal_loss import BFLWithLogits
from loss_functions.dice_loss import BDLWithLogits

PROJECT_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = "/dtu/datasets1/02516"
PH2_DATA_DIR = os.path.join(DATA_DIR, "PH2_Dataset_images")
DRIVE_DIR = os.path.join(DATA_DIR, "DRIVE")

def test_experiment(epochs=10):
    """Experiment to test the Code using the TestCNN model"""
    train_transform = random_transform(size=512, horizontal=True, vertical=True, rotation=True)
    test_transform = base_transform(size=512)
    dm = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=False, data_path=PH2_DATA_DIR, batch_size=8)
    # dm = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=True, data_path=DRIVE_DIR, batch_size=8)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        UNetDeconv
    ]

    description = [
        "Testing"
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}},
    ]

    # Loss functions that are working:
    # Binary Cross Entropy: torch.nn.BCEWithLogitsLoss()
    # Focal Loss: BFLWithLogits()
    # Binary Cross Entropy with Weights: torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).cuda())
    # Binary Dice Loss: BDLWithLogits()
    criterion_functions = [
        BFLWithLogits()
    ]

    epochs = [epochs]

    print(f"Training on dataset {dm.data_path}")
    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description, criterion_functions)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"), dm.data_path)