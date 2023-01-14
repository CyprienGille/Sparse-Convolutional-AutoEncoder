from pathlib import Path
from typing import Tuple

import numpy as np
import torch as T
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os


class ImageFolder720p(Dataset):
    """
    Image shape is (720, 1280, 3) --> (768, 1280, 3) --> 6x10 128x128 patches
    """

    def __init__(self, root: str, files_list=None):
        if files_list is None:
            self.files = sorted(Path(root).iterdir())
        else:
            self.files = sorted([Path(root) / k for k in files_list])

    def __getitem__(self, index: int) -> Tuple[T.Tensor, np.ndarray, Tuple[int, int]]:
        path = str(self.files[index % len(self.files)])
        img = np.array(Image.open(path))

        pad = ((24, 24), (0, 0), (0, 0))

        # img = np.pad(img, pad, 'constant', constant_values=0) / 255
        img = np.pad(img, pad, mode="edge") / 255.0

        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()

        patches = np.reshape(img, (3, 6, 128, 10, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))

        return img, patches, (6, 10)

    def __len__(self):
        return len(self.files)


class FlickrFolder(Dataset):
    """
    Image shape is (2048, 2048, 3) --> 16x16 128x128 patches
    """

    def __init__(self, root: str, files_list=None):
        if files_list is None:
            self.files = sorted(Path(root).iterdir())
        else:
            self.files = sorted([Path(root) / k for k in files_list])

    def __getitem__(self, index: int) -> Tuple[T.Tensor, np.ndarray, Tuple[int, int]]:
        path = str(self.files[index % len(self.files)])
        img = np.array(Image.open(path)).astype("float64")
        img /= 255
        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()

        patches = np.reshape(img, (3, 16, 128, 16, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))

        return img, patches, (16, 16)

    def __len__(self):
        return len(self.files)


class KodakFolder(Dataset):
    """
    Image shape is either (768, 512, 3) or (512, 768, 3) --> 6x4 or 4x6 128x128 patches
    """

    def __init__(self, root: str, files_list=None):
        if files_list is None:
            self.files = sorted(Path(root).iterdir())
        else:
            self.files = sorted([Path(root) / k for k in files_list])

    def __getitem__(self, index: int) -> Tuple[T.Tensor, np.ndarray, Tuple[int, int]]:
        path = str(self.files[index % len(self.files)])
        img = np.array(Image.open(path)).astype("float64")
        dims = img.shape
        if dims[0] == 768:
            pat_x, pat_y = 6, 4
        elif dims[0] == 512:
            pat_x, pat_y = 4, 6
        else:
            print(dims)
            pat_x, pat_y = -1, -1

        img /= 255
        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()

        patches = np.reshape(img, (3, pat_x, 128, pat_y, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))

        return img, patches, (pat_x, pat_y)

    def __len__(self):
        return len(self.files)
