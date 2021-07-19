import torchvision.transforms as transforms
import random
import cv2
import numpy as np


class SimCLRDataSetTransform:
    """
    Transform data to simCLR form
    Args:
        init:
            sigma (list(2, float), optional): gaussian blur parameters
            s (float, optional): colour jitter strength
            size (int): resize parameter
        GaussianBlur:
            image(tensor)
        call:
            image(Tensor): Image to tranform
    Returns:
        GaussianBlur:
            image(Tensor): Gaussian blurred image
        Augmentation:
            augmentations (torchvision.transforms.transforms.Compose): composed pytorch augmentations
        call:
            [q, k] (list(2, Tensor)): list of transformed image batches according to the simCLR
                                      framework
    """

    def __init__(self, s=1, size=28, strategy=None):
        # self.base_transform = base_transform
        self.s = s
        self.size = size
        self.base_transform = self._augmentation()
        self.strategy = strategy

    def _gaussian_blur(self, image):
        if random.uniform(0, 1) < 0.2:
            phi = random.uniform(0.1, 2)
            # ~10% of image size
            image = cv2.GaussianBlur(np.array(image), (21,21), phi)
        return image

    def _augmentation(self):
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        if self.strategy == 1:
            # no horizontal and vertical flip and rotation
            augmentations = transforms.Compose([transforms.RandomResizedCrop(size=self.size),
                                                # transforms.RandomHorizontalFlip(),
                                                # transforms.RandomRotation(degrees=(0, 360), fill=1),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                self._gaussian_blur,
                                                transforms.ToTensor()])
        elif self.strategy == 2:
            # no colour jitter and random greyscale
            augmentations = transforms.Compose([transforms.RandomResizedCrop(size=self.size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(degrees=(0, 360), fill=1),
                                                # transforms.RandomApply([color_jitter], p=0.8),
                                                # transforms.RandomGrayscale(p=0.2),
                                                self._gaussian_blur,
                                                transforms.ToTensor()])

        elif self.strategy == 3:
            # no gaussian blur or sobel filtering
            augmentations = transforms.Compose([transforms.RandomResizedCrop(size=self.size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(degrees=(0, 360), fill=1),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                # self._gaussian_blur,
                                                transforms.ToTensor()])
        else:
            augmentations = transforms.Compose([transforms.RandomResizedCrop(size=self.size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                self._gaussian_blur,
                                                transforms.ToTensor()])
        return augmentations

    def __call__(self, image):
        q = self.base_transform(image)
        k = self.base_transform(image)
        return [q, k]
