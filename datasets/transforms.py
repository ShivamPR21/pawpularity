from typing import Any, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.feature_extraction import image


class PawToTensor(object):
    def __init__(self,
                 image_size: Tuple[int, int] = None) -> None:
        self.image_transform = transforms.ToTensor()
        self.image_size = image_size

    def __call__(self,
                 pic: Image.Image) -> torch.Tensor:
        if self.image_size is not None:
            pic = pic.resize(self.image_size)

        return self.image_transform(pic)
