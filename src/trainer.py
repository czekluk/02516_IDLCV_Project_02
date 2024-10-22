import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from data.make_dataset import SegmentationDataModule
from data.custom_transforms import base_transform, random_transform
from typing import List
from torch.utils.data import DataLoader
import datetime
from metrics import accuracy, dice_overlap, intersection_over_union, sensitivity, specificity

class Trainer:
    
    def __init__(self, models: List[nn.Module], optimizer_functions: List[dict], 
                 epochs: int, train_loader: DataLoader, test_loader: DataLoader,
                 train_transform, description, criterion_functions) -> None:
        """
        Class for training different models with different optimizers and different numbers of epochs.
        
        Args:   models              -   list of models. The models are not instances but classes. example: [AlexNet, ResNet]
                optimizer_funcitons -   list of dictionaries specifying different optimizers.
                                        example: optimizers = [{"optimizer": torch.optim.Adam, "params": {"lr": 1e-3}}]
                epochs              -   list of different epochs to train. example: [10, 15]
                train_loader        -   torch.utils.data.DataLoader
                test_loader         -   torch.utils.data.DataLoader
        """
        assert len(models) == len(description)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterions = criterion_functions
        self.models = models
        self.optimizer_functions = optimizer_functions
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_transform = train_transform
        self.description = description
        print(f"Training on device: {self.device}")
    
    
    def train(self) -> List[dict]:
        """
        Train the different models, with different optimizers, with different number of epochs.
        
        Returns:    List of dictionaries representing different experiments.
                    The list is sorted in descending order based on the achieved accuracy
                    after the final epoch.
        """
        outputs = []
        count = 0
        for network in self.models:
            for criterion in self.criterions:
                for optimizer_config in self.optimizer_functions:
                    for epoch_no in self.epochs:
                        print("#########################################################")
                        print(f"Training model: {network.__name__}")
                        print(f"Description: {self.description[count]}")
                        print(f"Optimizer: {optimizer_config['optimizer'].__name__}")
                        print(f"Criterion: {criterion.__class__.__name__}")
                        print(f"Training for {epoch_no} epochs")
                        model = network()
                        out_dict = self._train_single_configuration(model, optimizer_config, epoch_no, criterion)
                        out_dict["description"] = self.description[count]
                        out_dict["timestamp"] = datetime.datetime.now()
                        out_dict["transform"] = self.train_transform
                        outputs.append(out_dict)
            count += 1
        outputs_sorted = sorted(outputs, key=lambda x: x['test_acc'][-1], reverse=True)
        return outputs_sorted
    
    
    def _train_single_configuration(self, model: nn.Module, optimizer_config: dict, num_epochs: int, criterion) -> dict:
        model.to(self.device)
        optimizer = optimizer_config["optimizer"](model.parameters(), **optimizer_config["params"])
        
        out_dict = {
            'model_name':       model.__class__.__name__,
            'description':      None,
            'timestamp':        None,
            'model':            model,
            'train_acc':        [],
            'test_acc':         [],
            'train_loss':       [],
            'test_loss':        [],
            'train_dice':       [],
            'test_dice':        [],
            'train_iou':        [],
            'test_iou':         [],
            'train_sensitivity':[],
            'test_sensitivity': [],
            'train_specificity':[],
            'test_specificity': [],
            'epochs':           num_epochs,
            'optimizer_config': optimizer_config,
            'criterion':        criterion.__class__.__name__,
            'transform':        None
            }
        
        for epoch in tqdm(range(num_epochs), unit='epoch'):
            model.train()
            train_loss = []
            train_acc = []
            train_dice = []
            train_iou = []
            train_sensitivity = []
            train_specificity = []
            
            for minibatch_no, (data, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.clone().detach().float().requires_grad_(True))
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
                sigmoid_output = torch.sigmoid(output)
                predicted = (sigmoid_output > 0.5).float()
                
                train_acc.append(accuracy(predicted, target).cpu().item())
                train_dice.append(dice_overlap(predicted, target).cpu().item())
                train_iou.append(intersection_over_union(predicted, target).cpu().item())
                train_sensitivity.append(sensitivity(predicted, target).cpu().item())
                train_specificity.append(specificity(predicted, target).cpu().item())
                
            
            test_loss = []
            test_acc = []
            test_dice = []
            test_iou = []
            test_sensitivity = []
            test_specificity = []
            model.eval()
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = model(data)
                test_loss.append(criterion(output, target.clone().detach().float().requires_grad_(True)).cpu().item())
                sigmoid_output = torch.sigmoid(output)
                predicted = (sigmoid_output > 0.5).float()
                test_acc.append(accuracy(predicted, target).cpu().item())
                test_dice.append(dice_overlap(predicted, target).cpu().item())
                test_iou.append(intersection_over_union(predicted, target).cpu().item())
                test_sensitivity.append(sensitivity(predicted, target).cpu().item())
                test_specificity.append(specificity(predicted, target).cpu().item())

            # Add entries output json
            out_dict['train_loss'].append(np.mean(train_loss))
            out_dict['test_loss'].append(np.mean(test_loss))

            out_dict['train_acc'].append(np.mean(train_acc))
            out_dict['test_acc'].append(np.mean(test_acc))

            out_dict['train_dice'].append(np.mean(train_dice))
            out_dict['test_dice'].append(np.mean(test_dice))

            out_dict['train_iou'].append(np.mean(train_iou))
            out_dict['test_iou'].append(np.mean(test_iou))

            out_dict['train_sensitivity'].append(np.mean(train_sensitivity))
            out_dict['test_sensitivity'].append(np.mean(test_sensitivity))

            out_dict['train_specificity'].append(np.mean(train_specificity))
            out_dict['test_specificity'].append(np.mean(test_specificity))

            # Print results of this epoch
            print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
                f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t",
                f"Dice train: {out_dict['train_dice'][-1]*100:.1f}\t test: {out_dict['test_dice'][-1]*100:.1f}\t",
                f"IoU train: {out_dict['train_iou'][-1]*100:.1f}\t test: {out_dict['test_iou'][-1]*100:.1f}")
            
        # Print final results
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
                f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t",
                f"Dice train: {out_dict['train_dice'][-1]*100:.1f}\t test: {out_dict['test_dice'][-1]*100:.1f}\t",
                f"IoU train: {out_dict['train_iou'][-1]*100:.1f}\t test: {out_dict['test_iou'][-1]*100:.1f}")
        
        return out_dict