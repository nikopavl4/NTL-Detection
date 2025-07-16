from types import SimpleNamespace
from typing import List, Optional, Union, Tuple

import torch

from ml.utils.train_utils import fit, predict


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) with support for Variational Auto-Encoders (VAE).

    Args:
        in_size (int): The size of the input features.
        layer_units (List[int]): A list of integers representing the number of units for each hidden layer.
        out_size (int): Size of the output features.
        init_weights (bool, optional): Whether to initialize the weights of the MLP. Defaults to True.
        vae (bool): Whether the MLP is used as a VAE. Defaults to False.
    """
    def __init__(self, in_size: int, layer_units: Optional[List[int]], out_size: int,
                 init_weights: Optional[bool] = True, vae: Optional[bool] = False):
        super(MLP, self).__init__()
        activation = torch.nn.ReLU()  # activation function

        layers = [torch.nn.Linear(in_size, layer_units[0]), activation]  # input layer
        for i in range(len(layer_units) - 1):  # hidden layers
            layers.append(torch.nn.Linear(layer_units[i], layer_units[i + 1]))
            layers.append(activation)
        layers.append(torch.nn.Linear(layer_units[len(layer_units) - 1], out_size))  # output layer
        if vae:
            layers.append(activation)
        self.MLP = torch.nn.Sequential(*layers)

        self.mu = None
        self.logvar = None
        if vae:
            self.mu = torch.nn.Linear(out_size, out_size)
            self.logvar = torch.nn.Linear(out_size, out_size)

        if init_weights:
            self.MLP.apply(self._init_weights)
            if vae:
                self.mu.apply(self._init_weights)
                self.logvar.apply(self._init_weights)

    def forward(self, x: torch.Tensor,
                sigmoid: Optional[bool] = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs forward propagation through the MLP.

        Args:
            x (torch.Tensor): The input tensor to the MLP.
            sigmoid (bool, optional): Whether to apply a sigmoid activation to the output.
        """
        x = self.MLP(x)
        if self.mu is not None and self.logvar is not None:
            mu = self.mu(x)
            logvar = self.logvar(x)
            return mu, logvar

        if sigmoid:
            x = torch.sigmoid(x)
        return x

    def _init_weights(self, module: torch.nn.Module):
        """
        Initializes the weights of the MLP.

        Args:
            module (torch.nn.Module): The torch model to initialize the weights.
        """
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def fit(self, args: SimpleNamespace) -> torch.nn.Module:
        """
        Trains the MLP using the fit function defined in train_utils.

        Args:
            args (SimpleNamespace): A dictionary of parameters for training.

        Returns:
            The trained MLP model.
        """
        model = fit(args.model, args.train_loader, args.test_loader, args.epochs,
                    args.optimizer, args.criterion, args.reconstruction, args.vae,
                    args.verbose, args.return_best, args.plot_history,
                    args.num_test_samples, args.device)
        return model

    def predict(self, args: SimpleNamespace) -> List[float]:
        """
        Makes predictions using the predict function defined in train_utils.

        Args:
            args (SimpleNamespace): A dictionary of parameters for making predictions.

        Returns:
            A list of evaluation metrics.
        """
        aux = predict(args.model, args.data_loader, args.criterion, args.reconstruction, args.num_samples, args.device)
        return aux
