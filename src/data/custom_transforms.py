import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image


class JointBaseTransform(torch.nn.Module):
    def __init__(self, size: int = 512):
        super().__init__()
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def forward(self, img1, img2):
        """
        Apply the same base transformations (resize, to tensor) to both input images.
        """
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2
    
    def __repr__(self):
        return f"J{self.__class__.__name__}(size={self.size})"



class JointRandomTransforms(torch.nn.Module):
    def __init__(self, size: int = 512, horizontal_p=0.5, vertical_p=0.5, rotation_degree=90, perspective_p=0.5):
        super().__init__()
        self.size = size
        self.horizontal_p = horizontal_p
        self.vertical_p = vertical_p
        self.rotation_degree = rotation_degree
        self.perspective_p = perspective_p
        
        self.base_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def forward(self, img1, img2):
        """
        Apply the same random transformations to both input and target images.
        """
        img1 = self.base_transform(img1)
        img2 = self.base_transform(img2)
        
        if torch.rand(1) < self.horizontal_p:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)

        if torch.rand(1) < self.vertical_p:
            img1 = F.vflip(img1)
            img2 = F.vflip(img2)

        if torch.rand(1) < self.perspective_p:
            startpoints, endpoints = T.RandomPerspective.get_params(img1.size(1), img1.size(0),distortion_scale=0.5)
            img1 = F.perspective(img1, startpoints, endpoints)
            img2 = F.perspective(img2, startpoints, endpoints)

        angle = T.RandomRotation.get_params([-self.rotation_degree, self.rotation_degree])
        img1 = F.rotate(img1, angle)
        img2 = F.rotate(img2, angle)

        return img1, img2

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(horizontal_p={self.horizontal_p}, "
                f"vertical_p={self.vertical_p}, rotation_degree={self.rotation_degree}, "
                f"perspective_p={self.perspective_p}, size={self.size})")


def base_transform(size: int = 512):
    return JointBaseTransform(size=size)


def random_transform(
    size: int = 512,
    horizontal: bool = False,
    horizontal_p: float = 0.5,
    vertical: bool = False,
    vertical_p: float = 0.5,
    rotation: bool = False,
    rotation_degree: int = 90,
    perspective: bool = False,
    perspective_p: float = 0.5,
):
    """Random transform chain for the image - resize, random horizontal flip, random vertical flip, random rotation, etc."""
    
    return JointRandomTransforms(
                size=size,
                horizontal_p=horizontal_p if horizontal else 0,
                vertical_p=vertical_p if vertical else 0,
                rotation_degree=rotation_degree if rotation else 0,
                perspective_p=perspective_p if perspective else 0
            )
