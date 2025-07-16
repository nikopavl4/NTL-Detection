import random
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class TorchDataset(Dataset):
    """
    PyTorch dataset that stores a set of features and labels.

    Args:
        X (torch.Tensor): A torch tensor of input features.
        y (torch.Tensor): A torch tensor of the corresponding labels.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Initialize the dataset with a set of features and labels.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the input features and the corresponding label for the specified sample.
        """
        return self.X[idx], self.y[idx]


class StratifiedSampler(Sampler):
    def __init__(self, targets: torch.Tensor, batch_size: int):
        super().__init__(targets)
        self.targets = targets
        self.batch_size = batch_size
        self.indices = self._stratify()

    def _stratify(self):
        class_indices: List[List[int]] = [[] for _ in range(2)]
        for idx, label in enumerate(self.targets):
            class_indices[label].append(idx)

        for indices in class_indices:
            random.shuffle(indices)

        num_batches = len(self.targets) // self.batch_size
        remainder = len(self.targets) % self.batch_size

        if remainder != 0:
            num_batches += 1

        if num_batches > len(class_indices[1]):
            raise ValueError(
                f"Cannot use a batch size of {self.batch_size}. The calculated number of batches is {num_batches} "
                f"and there are {len(class_indices[1])} samples for class 1. "
                f"Please consider using a larger batch size!")

        indices = [[] for _ in range(num_batches)]

        # distribute class 1 samples equally among batches
        class_1_batches = np.array_split(class_indices[1], num_batches)
        for i in range(num_batches):
            indices[i].extend(class_1_batches[i])

        # distribute class 0 samples
        class_0_pointer = 0
        for i in range(num_batches):
            while len(indices[i]) < self.batch_size and class_0_pointer < len(class_indices[0]):
                indices[i].append(class_indices[0][class_0_pointer])
                class_0_pointer += 1

        for i in range(len(indices)):
            random.shuffle(indices[i])

        return [idx for batch in indices for idx in batch]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
