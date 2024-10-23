import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
import torch
from models.encoder_decoder import EncDec_base, EncDecStride, EncDec_dropout, DilatedConvNet
from models.unet import UNetDeconv, UNetDilated
from data.custom_transforms import base_transform, random_transform
from data.make_dataset import SegmentationDataModule

PROJECT_BASE_DIR = os.path.dirname(os.path.abspath(''))
DEFAULT_PLOT_METRICS = ["train_acc", "test_acc", "test_dice", "test_iou", "test_sensitivity", "test_specificity"]
#DATA_DIR = "/dtu/datasets1/02516"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(PROJECT_BASE_DIR)), '02516')
PH2_DATA_DIR = os.path.join(DATA_DIR, "PH2_Dataset_images")
DRIVE_DIR = os.path.join(DATA_DIR, "DRIVE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Visualizer():
    def __init__(self):
        self.linestyles = ['solid', 'dashed', 'dotted', 'dashdot'] # Currently supports max. 4 different line styles
        self.device = device
    
    def plot_training_json(self, json_path, metrics=DEFAULT_PLOT_METRICS, cmap='Spectral', save_path=None, figsize=(8, 5)):
        """Plot the results of A SINGLE MODEL from a json file containing the results.
        
        Args:
            json_path (str, optional): Path to the json file containing the results. Defaults to "results/experiments_to_plot.json".
            metrics (list, optional): A list of the metric keys to plot from the json file. Defaults to ["train_acc", "test_acc", "test_dice", "test_iou", "test_sensitivity", "test_specificity"].
            cmap (str, optional): The colormap to use for the lines. Defaults to 'Spectral'.
            save_path (str, optional): The path to save the plot. Defaults to None (no save).
            figsize (tuple, optional): The size of the plot. Defaults to (8, 5).
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        if len(data) >= len(self.linestyles):
            raise ValueError("Visualizer currently only supports max. 4 different line styles. Please reduce the number of models to plot.")
        
        colors = plt.get_cmap(cmap, len(metrics))
        
        plt.figure(figsize=figsize)
        for i, entry in enumerate(data):
            for j, metric in enumerate(metrics):
                plt.plot(entry[metric], label=f"{entry['model_name']} {metric}", color=colors(j), linestyle=self.linestyles[i])
        plt.xlabel("Epoch")
        plt.xticks(range(max([len(entry["train_acc"]) for entry in data])))
        plt.ylabel("Accuracy")
        plt.legend()
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(save_path + f"training_{time}.png")
        plt.show()

    def plot_prediction_comparison(self, model, batch, save_path=None, figsize=(10, 5)):
        """Plot a comparison of the predicted and target images.

        Args:
            model: PyTorch model to use for prediction.
            batch: A batch of images and targets.
            save_path: Path to save the plot. Defaults to None (no save).
            figsize: Size of the plot. Defaults to (10, 5).
        """
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        with torch.no_grad():
            output = model(data)
        sigmoid_output = torch.sigmoid(output)
        predicted = (sigmoid_output > 0.5).float()
        
        f, (ax0, ax1) = plt.subplots(2, len(predicted), figsize=figsize)
        for i, (pred_img, tar_img) in enumerate(zip(predicted, target)):
            pred_img = pred_img.squeeze(0).cpu().numpy()
            tar_img = tar_img.squeeze(0).cpu().numpy()
            ax0[i].imshow(pred_img, cmap='gray', interpolation='none')
            ax0[i].set_title("Predicted")
            ax0[i].axis('off')
            ax1[i].imshow(tar_img, cmap='gray', interpolation='none')
            ax1[i].set_title("Target")
            ax1[i].axis('off')
        plt.tight_layout()
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            f.savefig(save_path + f"pred_comparison_{time}.png", bbox_inches='tight')
        plt.show()
        
    def plot_prediction_overlay_target(self, model, batch, save_path=None, figsize=(10, 3), cmap='cool'):
        """Plot a comparison of the predicted and target images by overlaying the prediction on the target.

        Args:
            model: PyTorch model to use for prediction.
            batch: A batch of images and targets.
            save_path: Path to save the plot. Defaults to None (no save).
            figsize: Size of the plot. Defaults to (10, 5).
        """
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        with torch.no_grad():
            output = model(data)
        sigmoid_output = torch.sigmoid(output)
        predicted = (sigmoid_output > 0.5).float()
        
        # Plot overlayed images
        f, ax = plt.subplots(1, len(predicted), figsize=figsize)
        for i, (pred_img, tar_img) in enumerate(zip(predicted, target)):
            pred_img = pred_img.squeeze(0).cpu().numpy()
            tar_img = tar_img.squeeze(0).cpu().numpy()
            masked = np.ma.masked_where(pred_img == 0, pred_img)
            
            ax[i].imshow(tar_img, cmap='gray', interpolation='none')
            ax[i].imshow(masked, alpha=0.6, cmap=cmap, vmin=0.99, vmax=1.,interpolation='none')
            ax[i].axis('off')

        # Add color legend
        pred_color = plt.get_cmap(cmap)(0.99)
        tar_color = plt.get_cmap('grey')(0.9)
        plt.plot(0, 0, "-", color=tar_color, label="Target")
        plt.plot(0, 0, "-", color=pred_color, label="Prediction")
        f.legend(loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05))
        
        # Save and show plot
        plt.tight_layout()
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            f.savefig(save_path + f"pred_overlay_target_{time}.png", bbox_inches='tight')
        plt.show()
        
    def plot_prediction_overlay_data(self, model, batch, save_path=None, figsize=(10, 3), cmap='cool'):
        """Plot a comparison of the predicted and data images by overlaying the prediction on the data.

        Args:
            model: PyTorch model to use for prediction.
            batch: A batch of images and targets.
            save_path: Path to save the plot. Defaults to None (no save).
            figsize: Size of the plot. Defaults to (10, 5).
        """
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        with torch.no_grad():
            output = model(data)
        sigmoid_output = torch.sigmoid(output)
        predicted = (sigmoid_output > 0.5).float()
        
        # Plot overlayed images
        f, ax = plt.subplots(1, len(predicted), figsize=figsize)
        for i, (pred_img, data_img) in enumerate(zip(predicted, data)):
            pred_img = pred_img.squeeze(0).cpu().numpy()
            data_img = data[i].squeeze(0).cpu().numpy()
            data_img = np.moveaxis(data_img, 0, -1) # move channel axis to last for plotting
            masked = np.ma.masked_where(pred_img == 0, pred_img)
            
            ax[i].imshow(data_img, cmap='gray', interpolation='none')
            ax[i].imshow(masked, alpha=0.2, cmap=cmap, vmin=0.99, vmax=1.,interpolation='none')
            ax[i].axis('off')

        # Add color legend
        pred_color = plt.get_cmap(cmap)(0.99)
        plt.plot(0, 0, "-", color=pred_color, label="Prediction", alpha=0.5)
        f.legend(loc='lower center', ncol=1, fontsize=10, bbox_to_anchor=(0.5, -0.05))
        plt.subplots_adjust(bottom=0.1)
        
        # Save and show plot
        plt.tight_layout()
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            f.savefig(save_path + f"pred_overlay_data_{time}.png", bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Init class
    visualizer = Visualizer()
    json_path = os.path.join(PROJECT_BASE_DIR, "results/experiments_to_plot.json")
    metrics_to_plot = ["train_acc", "test_acc"]
    save_path = os.path.join(PROJECT_BASE_DIR, "results/figures/")

    # Load model
    model = EncDec_base()
    model_path = os.path.join(PROJECT_BASE_DIR, "results/saved_models/EncDec_base-2024-10-21_0-51-51-0.8548-EncDec_base.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    # Get data
    test_transform = base_transform(size=512)
    dm = SegmentationDataModule(test_transform=test_transform, drive=False, data_path=PH2_DATA_DIR, batch_size=4)
    #dm = SegmentationDataModule(test_transform=test_transform, drive=True, data_path=DRIVE_DIR, batch_size=4)
    testloader = dm.test_dataloader()
    batch = next(iter(testloader))

    # Plot training results
    visualizer.plot_training_json(json_path=json_path, save_path=save_path, metrics=metrics_to_plot)
    visualizer.plot_prediction_comparison(model, batch, save_path=save_path)
    visualizer.plot_prediction_overlay_target(model, batch, save_path=save_path)
    visualizer.plot_prediction_overlay_data(model, batch, save_path=save_path)
