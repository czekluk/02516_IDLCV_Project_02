import os
import glob
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from data.custom_transforms import random_transform, base_transform
from data.weak_labels_creator import WeakLabelsCreator

from torch.utils.data import DataLoader, Dataset

PROJECT_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = "/dtu/datasets1/02516"
PH2_DATA_DIR = os.path.join(DATA_DIR, "PH2_Dataset_images")
DRIVE_DIR = os.path.join(DATA_DIR, "DRIVE")


class SegmentationDataset(Dataset):
    def __init__(
        self, train: bool, transform=transforms.ToTensor(), data_path=DATA_DIR, drive=True, train_split=0.8, 
        weak_annotation=False, point_level_strategy="random", num_points_per_label=10
    ):
        "Initialization"
        self.transform = transform
        self.weak_annotations = weak_annotation
        if self.weak_annotations:
            self.weak_labels_creator = WeakLabelsCreator(num_points_per_label, point_level_strategy)
        if drive:
            self.init_DRIVE(train, data_path, train_split)
        else:
            self.init_PH2(train, data_path, train_split)


    def init_DRIVE(self, train: bool, data_path: str, train_split: float):
        self.input_paths = []
        self.target_paths = []
        dataset_type_path = os.path.join(data_path, "training")
        for sub_folder_name in ['images', '1st_manual']:
            sub_folder_path = os.path.join(dataset_type_path, sub_folder_name)
            if 'images' in sub_folder_name:
                for image_file_name in sorted(os.listdir(sub_folder_path)):
                    image_path = os.path.join(sub_folder_path, image_file_name)
                    self.input_paths.append(image_path)
            elif '1st_manual' in sub_folder_name:
                for manual_file_name in sorted(os.listdir(sub_folder_path)):
                    manual_path = os.path.join(sub_folder_path, manual_file_name)
                    self.target_paths.append(manual_path)
        
        number_of_samples = len(self.input_paths)
        train_samples_number = int(number_of_samples * train_split)
        if train:
            self.input_paths = self.input_paths[:train_samples_number]
            self.target_paths = self.target_paths[:train_samples_number]
        else:
            self.input_paths = self.input_paths[train_samples_number:]
            self.target_paths = self.target_paths[train_samples_number:]
    
    
    def init_PH2(self, train: bool, data_path: str, train_split: float):
        self.input_paths = []
        self.target_paths = []
        
        data_folders = sorted(os.listdir(data_path))
        number_of_samples = len(data_folders)
        train_samples_number = int(number_of_samples * train_split)
        train_examples_folders = data_folders[0:train_samples_number]
        test_examples_folders = data_folders[train_samples_number:]
        if train:
            samples = train_examples_folders
        else:
            samples = test_examples_folders
            
        for folder_name in samples:
            folder_path = os.path.join(data_path, folder_name)
            for sub_folder_name in os.listdir(folder_path):
                sub_folder_path = os.path.join(folder_path, sub_folder_name)
                if 'Dermoscopic_Image' in sub_folder_name:
                    dermoscopic_image_path = os.path.join(sub_folder_path, os.listdir(sub_folder_path)[0])
                    self.input_paths.append(dermoscopic_image_path)
                elif 'lesion' in sub_folder_name:
                    lesion_image_path = os.path.join(sub_folder_path, os.listdir(sub_folder_path)[0])
                    self.target_paths.append(lesion_image_path)
                else:
                    # ignore the roi folder
                    continue
    
    def __len__(self):
        "Returns the total number of samples"
        return len(self.input_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        input_path = self.input_paths[idx]
        target_path = self.target_paths[idx]
        
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("L")
        
        # apply the same transform to both input and target
        x, y = self.transform(input_image, target_image)
        
        target_array = np.array(y)
        binary_target = (target_array > 0).astype(np.uint8)
        binary_target_tensor = torch.from_numpy(binary_target).float()
        if self.weak_annotations:
            return self.get_weakly_annotated(x, binary_target)
        
        return x, binary_target_tensor
    
    def get_weakly_annotated(self, x, binary_target):
        weakly_annotated = self.weak_labels_creator.create_points(np.squeeze(binary_target))
        return x, torch.from_numpy(binary_target).float(), weakly_annotated


class SegmentationDataModule:
    def __init__(
        self,
        data_path=DRIVE_DIR,
        batch_size: int = 16,
        train_transform=transforms.ToTensor(),
        test_transform=transforms.ToTensor(),
        drive=True,
        train_split=0.8,
        weak_annotations=False,
        point_level_strategy="random", 
        num_points_per_label=10
    ):
        """Custom data module class for the DRIVE and PH2 datasets. Used for
        loading of data, train/test splitting and constructing dataloaders.

        Args:
            data_path: Path to the data directory. Default: /dtu/datasets1/02516/DRIVE
            batch_size (int, optional): Batch size for the dataloaders. Defaults to 16.
            train_transform (_type_, optional): Transform to apply to train data. Defaults to transforms.ToTensor().
            test_transform (_type_, optional): Transform to apply to test data. Defaults to transforms.ToTensor().
            drive (bool, optional): Flag to use either DRIVE or PH2 dataset. Defaults to DRIVE dataset (use True for DRIVE, False for PH2)
            train_split (float, optional): Percentage of data points to use in train set. The rest will be used in test set. Defaults to 0.8.
            weak_annotations (bool, optional): If True, it will use weak annotations for train_dataset. Default False.
            point_level_strategy (string, optional): Strategy to use for weak annotations.
            num_points_per_label (int, optional): Number of points to be used for interest object and background in the weak labeling.
        """
        assert type(data_path)==str, "data_path needs to be a string"
        assert train_split < 1, "train_split needs to be less than 1"
        assert train_split > 0, "train_split needs to be more than 0"
        self.drive=drive
        self.batch_size = batch_size
        self.data_path = data_path
        self.point_level_strategy = point_level_strategy
        self.train_dataset = SegmentationDataset(
            train=True, transform=train_transform, data_path=data_path, drive=drive, train_split=train_split,
            weak_annotation=weak_annotations, point_level_strategy=point_level_strategy, num_points_per_label=num_points_per_label
        )
        self.test_dataset = SegmentationDataset(
            train=False, transform=test_transform, data_path=data_path, drive=drive, train_split=train_split,
            weak_annotation=weak_annotations, point_level_strategy=point_level_strategy, num_points_per_label=num_points_per_label
        )
        
    def train_dataloader(self, shuffle=True) -> DataLoader:
        """Return the training dataloader

        Args:
            shuffle (bool, optional, Default: True): Whether to shuffle the dataset. Defaults to True.

        Returns:
            DataLoader: torch.utils.data.DataLoader
        """
        self.trainloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=3,
        )
        return self.trainloader

    def test_dataloader(self, shuffle=False):
        """Return the test dataloader

        Args:
            shuffle (bool, optional, Default: False): Whether to shuffle the dataset. Defaults to True.

        Returns:
            DataLoader: torch.utils.data.DataLoader
        """
        self.testloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=3,
        )
        return self.testloader

    def get_training_examples(self):
        """Return the first batch of training examples

        Returns:
            images: torch.Tensor
            labels: torch.Tensor
        """
        images, labels = next(iter(self.trainloader))
        return images, labels

    def get_test_examples(self):
        """Return the first batch of test examples

        Returns:
            images: torch.Tensor
            labels: torch.Tensor
        """
        images, labels = next(iter(self.testloader))
        return images, labels

    def plot_examples(self):
        """Plot the first batch of training examples"""
        images, labels = next(iter(self.trainloader))
        image, label = images[0], labels[0]
        image = image.permute(1, 2, 0).numpy()
        label = label.squeeze().numpy()
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(label, cmap='gray')
        plt.title('Target')
        plt.axis('off')

        plt.show()
    
    def plot_weakly_labeled_examples(self):
        """Plot the first batch of weakly labeled training examples"""
        images, labels, point_supervision = next(iter(self.trainloader))  # Assuming point supervision is returned from DataLoader
        image, label, point_label = images[0], labels[0], point_supervision[0]
        
        # Convert to numpy for plotting
        image = image.permute(1, 2, 0).numpy()
        label = label.squeeze().numpy()
        point_label = point_label.squeeze().numpy()

        # Overlay points on the original mask
        foreground_points = np.argwhere(point_label == 1)
        background_points = np.argwhere(point_label == 0)

        # Plot original image, full label, and weakly labeled image
        plt.figure(figsize=(15, 5))

        # Original input image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')

        # Full target label
        plt.subplot(1, 3, 2)
        plt.imshow(label, cmap='gray')
        plt.title('Full Target Label')
        plt.axis('off')

        # Weakly labeled mask with point clicks
        plt.subplot(1, 3, 3)
        plt.imshow(label, cmap='gray')

        # Plot foreground points in green
        for point in foreground_points:
            plt.scatter(point[1], point[0], c='green', s=50, label='Positive Click' if point[0] == foreground_points[0][0] else "")

        # Plot background points in red
        for point in background_points:
            plt.scatter(point[1], point[0], c='red', s=50, label='Negative Click' if point[0] == background_points[0][0] else "")

        plt.title(f'Weakly Labeled, {self.point_level_strategy} strategy (Positive: Green, Negative: Red)')
        plt.axis('off')

        # Display the legend only once (to avoid duplicate labels)
        plt.legend(loc="upper right")

        plt.show()
            
    def __repr__(self):
        return (
            f"Segmentation DataModule with batch size {self.batch_size}\n"
            + f" Training dataset: {len(self.train_dataset)} samples\n"
            + f" Test dataset: {len(self.test_dataset)} samples"
        )
    
    def get_trainset_size(self):
        return len(self.train_dataset)

    def get_testset_size(self):
        return len(self.test_dataset)


def test_drive():
    data_dir = os.path.join(PROJECT_BASE_DIR, "data")
    drive_dir = os.path.join(data_dir, "DRIVE")
    print("Data directory: ", drive_dir)

    img_size = 512
    train_transform = random_transform(size=img_size, horizontal=True, rotation=True)
    test_transform = base_transform(size=img_size)

    dm_drive = SegmentationDataModule(
        data_path=drive_dir, train_transform=train_transform, test_transform=test_transform, drive=True
    )
    print(dm_drive)
    trainloader = dm_drive.train_dataloader()
    testloader = dm_drive.test_dataloader()

    dm_drive.plot_examples()


def test_ph2():
    data_dir = os.path.join(PROJECT_BASE_DIR, "data")
    ph2_data_dir = os.path.join(data_dir, "PH2_Dataset_images")
    print("Data directory: ", ph2_data_dir)

    img_size = 512
    train_transform = random_transform(size=img_size, horizontal=True, rotation=True)
    test_transform = base_transform(size=img_size)

    dm_ph2 = SegmentationDataModule(
        data_path=ph2_data_dir, train_transform=train_transform, test_transform=test_transform, drive=False
    )
    print(dm_ph2)
    trainloader = dm_ph2.train_dataloader()
    testloader = dm_ph2.test_dataloader()

    dm_ph2.plot_examples()


def test_ph2_weakly_annotated():
    data_dir = os.path.join(PROJECT_BASE_DIR, "data")
    ph2_data_dir = os.path.join(data_dir, "PH2_Dataset_images")
    print("Data directory: ", ph2_data_dir)

    img_size = 512
    train_transform = random_transform(size=img_size, horizontal=True, rotation=True)
    test_transform = base_transform(size=img_size)

    dm_ph2 = SegmentationDataModule(
        data_path=ph2_data_dir, train_transform=train_transform, test_transform=test_transform, drive=False,
        weak_annotations=True, point_level_strategy="extreme_clicks", num_points_per_label=10
    )
    
    trainloader = dm_ph2.train_dataloader()
    testloader = dm_ph2.test_dataloader()

    dm_ph2.plot_weakly_labeled_examples()


if __name__ == "__main__":
    test_ph2_weakly_annotated()
    test_ph2()
    test_drive()
