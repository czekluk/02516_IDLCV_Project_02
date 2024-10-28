import torch
import os
import json
import argparse
from copy import deepcopy

from trainer import Trainer
from utils import save_results
from data.custom_transforms import base_transform, random_transform
from data.make_dataset import SegmentationDataModule
from models.encoder_decoder import EncDec_base, EncDecStride, EncDec_dropout, DilatedConvNet, EncDec_batchnorm_dropout
from models.unet import UNetDeconv, UNetDilated
from loss_functions.focal_loss import BFLWithLogits
from loss_functions.dice_loss import BDLWithLogits
from loss_functions.point_level_loss import PLLWithLogits

from copy import deepcopy
PROJECT_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = "/dtu/datasets1/02516"
PH2_DATA_DIR = os.path.join(DATA_DIR, "PH2_Dataset_images")
DRIVE_DIR = os.path.join(DATA_DIR, "DRIVE")

# Load configuration from config.json
max_epochs = 100
with open(os.path.join(PROJECT_BASE_DIR, 'set_epochs.json'), 'r') as f:
    config = json.load(f)

max_epochs = config.get("max_epochs", 100)  # Default to 100 if not found

# Define a function to get the number of epochs based on the dataset type and class
def get_epochs(dataset_type, class_name):
    dataset_config = config.get(dataset_type, {})
    class_proportions = dataset_config.get("class_proportions", {})
    proportion = class_proportions.get(class_name, dataset_config.get("default", 1.0))
    epochs = int(max_epochs * proportion)
    print(f"Epochs for {dataset_type} {class_name}: {epochs}")
    return epochs

# Define a function to get the criterion functions and their descriptions
def get_criterion_functions():
    return [
        (PLLWithLogits(),"PLLWithLogits")
        # (torch.nn.BCEWithLogitsLoss(), "BCEWithLogitsLoss")
        # (BFLWithLogits(), "BFLWithLogits")
        # # (torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).cuda()), "BCEWithLogitsLoss with pos_weight"),
        # # (BDLWithLogits(), "BDLWithLogits")
    ]

def test_experiment():
    """Experiment to test the Code using the TestCNN model"""
    train_transform = random_transform(size=512, horizontal=True, vertical=True, rotation=True)
    test_transform = base_transform(size=512)
    dm_ph2 = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=False, data_path=PH2_DATA_DIR, batch_size=8, 
                                weak_annotations=True, point_level_strategy="central_clicks", num_points_per_label=10)
    # dm_ph2 = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=False, data_path=PH2_DATA_DIR, batch_size=8)
    dm_drive = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=True, data_path=DRIVE_DIR, batch_size=8, 
                                weak_annotations=True, point_level_strategy="central_clicks", num_points_per_label=10)
    # dm_drive = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=True, data_path=DRIVE_DIR, batch_size=8)
    # trainloader = dm_ph2.train_dataloader()
    # testloader = dm_ph2.test_dataloader()

    optimizers_gl = [
        {"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}},
    ]
    criterion_functions = get_criterion_functions()

    enc_dec_models = [
        EncDec_batchnorm_dropout,
        EncDec_base,
        EncDecStride,
        EncDec_dropout,
        DilatedConvNet
    ]

    enc_dec_descriptions = [
        "EncDec_batchnorm_dropout",
        "EncDec_base",
        "EncDecStride",
        "EncDec_dropout",
        "DilatedConvNet"
    ]

    unet_models = [
        UNetDeconv,
        UNetDilated
    ]

    unet_descriptions = [
        "UNetDeconv",
        "UNetDilated"
    ]

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Control which dataset and models to run")
    parser.add_argument("--dataset", choices=["drive", "ph2", "both"], default="both", help="Dataset to use")
    parser.add_argument("--models", choices=["encdec", "unet", "both"], default="both", help="Models to use")
    args = parser.parse_args()
    print("received args:", args)
    # Train and save results for encoder-decoder models
    if args.models in ["encdec", "both"]:
        for dm_instance in [dm_drive, dm_ph2]:
            dataset_type = "drive" if dm_instance.drive else "ph2"
            if args.dataset in [dataset_type, "both"]:
                for criterion in criterion_functions:
                    optimizers = deepcopy(optimizers_gl)
                    epochs_list = [get_epochs(dataset_type, model) for model in enc_dec_descriptions]
                    trainer = Trainer(
                        enc_dec_models, optimizers, epochs_list, dm_instance.train_dataloader(), dm_instance.test_dataloader(), train_transform, enc_dec_descriptions, [criterion]
                    )
                    outputs = trainer.train()
                    results_path = os.path.join(PROJECT_BASE_DIR, f"results/experiments_encdec_{dataset_type}.json")
                    print(f"Saving results to {results_path}")
                    save_results(outputs, results_path, dm_instance.data_path)

    # Train and save results for UNet models
    if args.models in ["unet", "both"]:
        for dm_instance in [dm_drive, dm_ph2]:
            dataset_type = "drive" if dm_instance.drive else "ph2"
            if args.dataset in [dataset_type, "both"]:
                for criterion in criterion_functions:
                    optimizers = deepcopy(optimizers_gl)
                    epochs_list = [get_epochs(dataset_type, model) for model in unet_descriptions]
                    trainer = Trainer(
                        unet_models, optimizers, epochs_list, dm_instance.train_dataloader(), dm_instance.test_dataloader(), train_transform, unet_descriptions, [criterion]
                    )
                    outputs = trainer.train()
                    results_path = os.path.join(PROJECT_BASE_DIR, f"results/experiments_unet_{dataset_type}.json")
                    print(f"Saving results to {results_path}")
                    save_results(outputs, results_path, dm_instance.data_path)
    print("Experiment completed")

if __name__ == "__main__":
    # test_experiment()
    results_dir = os.path.join(PROJECT_BASE_DIR, "results")
    output_csv = os.path.join(PROJECT_BASE_DIR, "results_summary.csv")