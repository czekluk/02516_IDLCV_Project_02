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
import csv
from copy import deepcopy
from decimal import Decimal

PROJECT_BASE_DIR = os.path.dirname("/zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_02/")
DEFAULT_PLOT_METRICS = ["test_dice", "test_iou", "test_sensitivity", "test_specificity"]

DATA_DIR = "/dtu/datasets1/02516"
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
        
       
        # print(f"Warning: More than {len(self.linestyles)} models to plot. Reducing to the first {len(self.linestyles)} models.")
        # Filter to include only the best UNet and the best non-UNet models
        unet_models = [entry for entry in data if entry['model_name'].lower().startswith('unet')]
        non_unet_models = [entry for entry in data if 'unet' not in entry['model_name'].lower()]
        
        best_non_unet_model=None
        best_unet_model=None
        if unet_models:
            best_unet_model = max(unet_models, key=lambda x: max(x.get('test_iou', [0])))
        if non_unet_models:
            best_non_unet_model = max(non_unet_models, key=lambda x: max(x.get('test_iou', [0])))
        data = [model for model in [best_unet_model, best_non_unet_model] if model is not None]
        colors = plt.get_cmap(cmap, len(metrics))
        
        plt.figure(figsize=figsize)
        for i, entry in enumerate(data):
            for j, metric in enumerate(metrics):
                model_name = entry['model_name']
                existing_labels = [line.get_label() for line in plt.gca().get_lines()]
                index = 1
                while f"{model_name} {metric}" in existing_labels:
                    model_name = f"{entry['model_name']}_{index}"
                    index += 1
                plt.plot(entry[metric], label=f"{model_name} {metric}", color=colors(j), linestyle=self.linestyles[i])
        plt.xlabel("Epoch")
        max_epochs = max([len(entry["train_acc"]) for entry in data])
        plt.xticks(np.linspace(0, max_epochs, 5, dtype=int))
        plt.ylabel("Metrics")
        plt.legend()
        dataset_type = "drive" if "drive" in json_path else "ph2"
        plt.title(f"Training Metrics for the {dataset_type} dataset")
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(save_path + f"training_{time}.png")
        plt.show()

    def plot_all_json_files(self, json_files, metrics=DEFAULT_PLOT_METRICS, cmap='Spectral', save_dir=None, figsize=(8, 5)):
        """Plot the results from multiple json files containing the results.
        
        Args:
            json_files (list): List of paths to the json files containing the results.
            metrics (list, optional): A list of the metric keys to plot from the json files. Defaults to ["train_acc", "test_acc", "test_dice", "test_iou", "test_sensitivity", "test_specificity"].
            cmap (str, optional): The colormap to use for the lines. Defaults to 'Spectral'.
            save_dir (str, optional): The directory to save the plots. Defaults to None (no save).
            figsize (tuple, optional): The size of the plot. Defaults to (8, 5).
        """
        united_data = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                united_data.extend(data)
        
        drive_data = []
        ph2_data = []
        
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                if "drive" in json_file:
                    drive_data.extend(data)
                elif "ph2" in json_file:
                    ph2_data.extend(data)

        drive_json_path = os.path.join(PROJECT_BASE_DIR, "results", "experiments_drive_united.json")
        ph2_json_path = os.path.join(PROJECT_BASE_DIR, "results", "experiments_ph2_united.json")

        with open(drive_json_path, "w") as f:
            json.dump(drive_data, f, indent=4)

        with open(ph2_json_path, "w") as f:
            json.dump(ph2_data, f, indent=4)
        json_files = []
        json_files.append(drive_json_path)
        json_files.append(ph2_json_path)
        for json_file in json_files:
            if "drive" in json_file:
                dataset_type = "drive"
            elif "ph2" in json_file:
                dataset_type = "ph2"
            else:
                dataset_type = "united"
            save_path = os.path.join(save_dir, f"{dataset_type}_united") if save_dir else None
            self.plot_training_json(json_file, metrics=metrics, cmap=cmap, save_path=save_path, figsize=figsize)

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
        
    def plot_weakly_labeled_prediction(self, model, batch, save_path=None, figsize=(10, 3), cmap='cool'):
        """Plot a comparison of the predicted and data images by overlaying the prediction on the weakly labeled points

        Args:
            model: PyTorch model to use for prediction.
            batch: A batch of images, targets, and supervised points.
            save_path: Path to save the plot. Defaults to None (no save).
            figsize: Size of the plot. Defaults to (10, 5).
        """
        data, target, points = batch
        data, target = data.to(self.device), target.to(self.device)
        points = points.to(self.device).squeeze().numpy()
        
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
            
            # Plot weakly labeled points
            foreground_points = np.argwhere(points[i] == 1)
            background_points = np.argwhere(points[i] == 0)
            
            # Plot foreground points in green
            for j, point in enumerate(foreground_points):
                ax[i].scatter(point[1], point[0], c='green', s=50, label='Positive Click' if (j == 0) and (i == 0) else None)

            # Plot background points in red
            for j, point in enumerate(background_points):
                ax[i].scatter(point[1], point[0], c='red', s=50, label='Positive Click' if (j == 0) and (i == 0) else None)
                
            ax[i].axis('off')

        # Add color legend
        pred_color = plt.get_cmap(cmap)(0.99)
        tar_color = plt.get_cmap('grey')(0.9)
        plt.plot(0, 0, "-", color=tar_color, label="Target")
        plt.plot(0, 0, "-", color=pred_color, label="Prediction")
        f.legend(loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.05))
        
        # Save and show plot
        plt.tight_layout()
        if save_path:
            time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            f.savefig(save_path + f"pred_overlay_target_{time}.png", bbox_inches='tight')
        plt.show()
    
def get_best_model(summary_path):
    with open(summary_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        best_model = None
        best_diou = -float('inf')
        for row in reader:
            model_name, test_diou = row[0], float(row[5])  # Assuming Test IoU is the 7th column
            if test_diou > best_diou:
                best_model = model_name
                best_diou = test_diou
    return best_model, best_diou
if __name__ == "__main__":
    # Init class
    
    json_files = [
        os.path.join(PROJECT_BASE_DIR, "results/experiments_encdec_drive.json"),
        os.path.join(PROJECT_BASE_DIR, "results/experiments_encdec_ph2.json"),
        os.path.join(PROJECT_BASE_DIR, "results/experiments_unet_drive.json"),
        os.path.join(PROJECT_BASE_DIR, "results/experiments_unet_ph2.json")
    ]
    metrics_to_plot = ["test_acc", "test_dice", "test_iou", "test_sensitivity", "test_specificity"]
    # metrics_to_plot = ["test_iou"]

    save_dir = os.path.join(PROJECT_BASE_DIR, "results/figures/")
    
    visualizer = Visualizer()
    # # Plot training results for all JSON files
    visualizer.plot_all_json_files(json_files, metrics=metrics_to_plot, save_dir=save_dir)


    # Load summary files
    # Identify the best models based on test_iou
    # Collect dIoU values from summary files
    PROJECT_BASE_DIR = os.path.dirname("/zhome/25/a/202562/intro_deep_learning_in_computer_vision/02516_IDLCV_Project_02/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load summary files
    summary_files = [
        os.path.join(PROJECT_BASE_DIR, "results/summary_drive.csv"),
        os.path.join(PROJECT_BASE_DIR, "results/summary_ph2.csv")
]
    summary_diou = {}
    for summary_file in summary_files:
        dataset = "drive" if "drive" in summary_file else "ph2"
        with open(summary_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
           
            best_models = []
            done=True
            for row in reader:
                if done:
                    first_row = deepcopy(row)
                    done= False
                model_name, test_diou = row[0], float(row[5])
                best_models.append((model_name, test_diou))
            best_models = sorted(best_models, key=lambda x: x[1], reverse=True)
            for best_model, best_diou in best_models:
                if (best_model, dataset) not in summary_diou:
                    summary_diou[(best_model, dataset)] = []
                summary_diou[(best_model, dataset)].append(best_diou)
            # best_diou = float(first_row[5])
            # summary_diou[(best_model, dataset)] = round(best_diou, 3)
    # Reduce summary_diou to contain only the best dIoU from both the ph2 and drive datasets
    best_summary_diou = {}
    for dataset in ["drive", "ph2"]:
        best_model = max(
            ((model, max(diou_list)) for (model, ds), diou_list in summary_diou.items() if ds == dataset),
            key=lambda x: x[1],
            default=None
        )
        if best_model:
            best_summary_diou[(best_model[0], dataset)] = best_model[1]
    summary_diou = best_summary_diou
    # Identify the best models based on dIoU
    best_models = []

    saved_models_dir = os.path.join(PROJECT_BASE_DIR, "results/saved_models")
    saved_model_files = os.listdir(saved_models_dir)
    saved_model_files = [file for file in saved_model_files if not file.startswith('.git')]
    for model_file in saved_model_files:
        parts = model_file.split('-')
        model_name = parts[0]
        decimal_value= Decimal(parts[6])
        diou = round(decimal_value, 3)
        for (summary_model_name, dataset), summary_diou_value in summary_diou.items():
            summary_diou_value=Decimal(summary_diou_value)
            summary_diou_value = round(summary_diou_value, 3)
            if model_name == summary_model_name and diou == summary_diou_value:
                best_models.append((model_name, diou, model_file, dataset))

    # Paths to the best models
    best_drive_model_path = None
    best_ph2_model_path = None

    for model_name, diou, model_file, dataset in best_models:
        model_path = os.path.join(saved_models_dir, model_file)
        if dataset == "ph2":
            best_ph2_model_path = model_path
        elif dataset == "drive":
            best_drive_model_path = model_path
      

    # Print paths to the best models
    print(f"Best Drive Model Path: {best_drive_model_path}")
    print(f"Best PH2 Model Path: {best_ph2_model_path}")

    # Load models
    drive_model = torch.load(best_drive_model_path, map_location=device)
    drive_model_name = os.path.basename(best_drive_model_path).split('-')[0]
    drive_model = eval(drive_model_name)()
    drive_model.load_state_dict(torch.load(best_drive_model_path, map_location=device))
    drive_model.to(device)
    ph2_model = torch.load(best_ph2_model_path, map_location=device)
    ph2_model_name = os.path.basename(best_ph2_model_path).split('-')[0]
    ph2_model = eval(ph2_model_name)()
    ph2_model.load_state_dict(torch.load(best_ph2_model_path, map_location=device))
    ph2_model.to(device)
   
    train_transform = random_transform(size=512)
    test_transform = base_transform(size=512)
    # data_module_ph2 = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=False, data_path=PH2_DATA_DIR, batch_size=8)
    data_module_ph2 = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=False, data_path=PH2_DATA_DIR, batch_size=8, 
                                weak_annotations=True, point_level_strategy="central_clicks", num_points_per_label=10)
    # data_module_drive = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=True, data_path=DRIVE_DIR, batch_size=8)
    data_module_drive = SegmentationDataModule(train_transform=train_transform, test_transform=test_transform, drive=True, data_path=DRIVE_DIR, batch_size=8, 
                                weak_annotations=True, point_level_strategy="central_clicks", num_points_per_label=10)
    drive_batch = next(iter(data_module_drive.test_dataloader()))
    ph2_batch = next(iter(data_module_ph2.test_dataloader()))
    drive_batch = next(iter(data_module_drive.test_dataloader()))
    # Plot predictions
    # visualizer.plot_prediction_overlay_target(drive_model, drive_batch, save_path=os.path.join(PROJECT_BASE_DIR, "results"))
    # visualizer.plot_prediction_overlay_target(ph2_model, ph2_batch, save_path=os.path.join(PROJECT_BASE_DIR, "results"))
    # visualizer.plot_prediction_overlay_data(drive_model, drive_batch, save_path=os.path.join(PROJECT_BASE_DIR, "results"))
    # visualizer.plot_prediction_overlay_data(ph2_model, ph2_batch, save_path=os.path.join(PROJECT_BASE_DIR, "results"))
    visualizer.plot_weakly_labeled_prediction(drive_model, drive_batch, save_path=os.path.join(PROJECT_BASE_DIR, "results"), figsize=(10, 3), cmap='cool')
    visualizer.plot_weakly_labeled_prediction(ph2_model, ph2_batch, save_path=os.path.join(PROJECT_BASE_DIR, "results"), figsize=(10, 3), cmap='cool')


