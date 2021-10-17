from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
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

    def _load_suplimetry_data(self, index: int) -> pd.Series:
        sup_data = self.data.iloc[index, 1:-1]

        return sup_data.values

    def _load_target(self, index: int) -> np.float32:
        return np.float32(self.data.iloc[index, -1])

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        id = str(self.data.iloc[index, 0])
        image_data = self._load_image(id)
        sup_data = self._load_suplimetry_data(index)
        target = self._load_target(index)

        # if self.transforms is not None:
        #     image_data = self.transforms(image_data)

        # if self.transform is not None:
        #     image_data, sup_data, target = \
        #         self.transform(image_data), self.transform(sup_data), self.transform(target)

        return image_data, sup_data, target

    def __len__(self) -> int:
        return self.data.shape[0]
