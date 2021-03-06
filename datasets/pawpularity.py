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

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from cv2 import transform
from PIL import Image
from sklearn.feature_extraction import image
from torchvision.datasets import VisionDataset


class PawpularityPreditction(VisionDataset):

    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms=transforms,
                         transform=transform, target_transform=target_transform)

        self.data = pd.read_csv(f"{self.root}/train.csv")

    def _load_image(self, id: str) -> Image.Image:
        image_path = f"{self.root}/train/{id}.jpg"
        pet_image = Image.open(image_path).convert("RGB")

        return pet_image

    def _load_target(self, index: int) -> np.ndarray:
        return np.float32(self.data.iloc[index, 1:])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = str(self.data.iloc[index, 0])
        image_data = self._load_image(id)
        target = self._load_target(index)

        if self.transforms is not None:
            image_data, target = self.transforms(image_data, target)

        return image_data, target

    def __len__(self) -> int:
        return self.data.shape[0]
