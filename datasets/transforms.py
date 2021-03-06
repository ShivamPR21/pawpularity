'''
Copyright (C) 2021  Shivam Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

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
