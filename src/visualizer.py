import json
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from datetime import datetime
import os
import seaborn as sns
import numpy as np
import torch

from tqdm import tqdm

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PLOT_METRICS = ["train_acc", "test_acc", "test_dice", "test_iou", "test_sensitivity", "test_specificity"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Visualizer():
    def __init__(self):
        self.model = None
        self.linestyles = ['solid', 'dashed', 'dotted', 'dashdot'] # Currently supports max. 4 different line styles
    
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
        
    def load_model(self, model_path):
        raise NotImplementedError
        #self.model = torch.load(model_path, map_location=device)
        #self.model.eval()
    
    def plot_prediction(self, image):
        raise NotImplementedError
        #return self.model(image)
    
if __name__ == "__main__":
    # Init class
    visualizer = Visualizer()
    json_path = os.path.join(PROJECT_BASE_DIR, "results/experiments_to_plot.json")
    metrics_to_plot = ["train_acc", "test_acc"]
    #save_path = os.path.join(PROJECT_BASE_DIR, "results/figures/")

    # Plot training results
    visualizer.plot_training_json(json_path=json_path, save_path=None, metrics=metrics_to_plot)
    # TODO: plot prediction vs. ground truth
