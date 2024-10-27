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

class JointRandomCropTransforms(JointRandomTransforms):
    def __init__(self, size: int = 512, crop_size: int = 256, **kwargs):
        """
        Initialize the random crop transform.
        
        Args:
            size (int): The size to which the images will be resized after cropping.
            crop_size (int): The size of the random crop.
            num_crops (int): The number of crops to generate from each image.
            kwargs: Additional arguments for random transforms.
        """
        super().__init__(size=size, **kwargs)
        self.crop_size = crop_size

    def forward(self, img1, img2):
        """
        Apply random cropping and other transformations to both images.
        
        Args:
            img1 (PIL.Image or Tensor): The input image.
            img2 (PIL.Image or Tensor): The corresponding mask.
        
        Returns:
            (img1, img2): Transformed pair of images.
        """
        img1 = self.base_transform(img1)
        img2 = self.base_transform(img2)
        
        # Random crop
        i, j, h, w = T.RandomCrop.get_params(img1, output_size=(self.crop_size, self.crop_size))
        cropped_img1 = F.crop(img1, i, j, h, w)
        cropped_img2 = F.crop(img2, i, j, h, w)
        
        # Apply other random transformations
        if torch.rand(1) < self.horizontal_p:
            cropped_img1 = F.hflip(cropped_img1)
            cropped_img2 = F.hflip(cropped_img2)

        if torch.rand(1) < self.vertical_p:
            cropped_img1 = F.vflip(cropped_img1)
            cropped_img2 = F.vflip(cropped_img2)

        if torch.rand(1) < self.perspective_p:
            startpoints, endpoints = T.RandomPerspective.get_params(cropped_img1.size(1), cropped_img1.size(0), distortion_scale=0.5)
            cropped_img1 = F.perspective(cropped_img1, startpoints, endpoints)
            cropped_img2 = F.perspective(cropped_img2, startpoints, endpoints)

        angle = T.RandomRotation.get_params([-self.rotation_degree, self.rotation_degree])
        cropped_img1 = F.rotate(cropped_img1, angle)
        cropped_img2 = F.rotate(cropped_img2, angle)

        # Resize the crops back to the original size
        cropped_img1 = F.resize(cropped_img1, (self.size, self.size))
        cropped_img2 = F.resize(cropped_img2, (self.size, self.size))

        return cropped_img1, cropped_img2

    def __repr__(self):
        return (f"{self.__class__.__name__}(crop_size={self.crop_size} "
                f"horizontal_p={self.horizontal_p}, vertical_p={self.vertical_p}, "
                f"rotation_degree={self.rotation_degree}, perspective_p={self.perspective_p}, size={self.size})")


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

def random_crop_transform(
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
    
    return JointRandomCropTransforms(
                size=size,
                crop_size=size//2,
                horizontal_p=horizontal_p if horizontal else 0,
                vertical_p=vertical_p if vertical else 0,
                rotation_degree=rotation_degree if rotation else 0,
                perspective_p=perspective_p if perspective else 0
            )
