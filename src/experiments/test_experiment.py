import torch
import os

from src.trainer import Trainer
from src.utils import save_results
from data.custom_transforms import base_transform, random_transform
from data.make_dataset import SegmentationDataModule
from models.test_cnn import TestCNN

PROJECT_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

def test_experiment(epochs=10):
    """Experiment to test the Code using the TestCNN model"""
    train_transform = random_transform(normalize=True,size=256, rotation=True, perspective=True, random_erasing=True)
    test_transform = base_transform(normalize=True,size=256)
    dm = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform)
    trainloader = dm.train_dataloader()
    testloader = dm.test_dataloader()

    models = [
        TestCNN
    ]

    description = [
        "TestCNN",
    ]

    optimizers = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-4, "weight_decay": 1e-5}}
    ]

    epochs = [epochs]

    trainer = Trainer(models, optimizers, epochs, trainloader, testloader, train_transform, description)
    outputs = trainer.train()
    save_results(outputs, os.path.join(PROJECT_BASE_DIR, "results/experiments.json"))