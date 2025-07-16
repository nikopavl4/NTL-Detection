from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.utils.train_utils import fit, predict


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim):
        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x1 = self.fc(x)
        x2 = torch.sigmoid(self.gate(x))
        return x1 * x2


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers, n_glu):
        super(FeatureTransformer, self).__init__()
        self.shared_layers = shared_layers
        self.layer1 = GatedLinearUnit(input_dim)
        self.layers = nn.ModuleList([GatedLinearUnit(output_dim) for _ in range(n_glu - 1)])

    def forward(self, x):
        if self.shared_layers is not None:
            for layer in self.shared_layers:
                x = layer(x)
        x = self.layer1(x)
        for layer in self.layers:
            x = x + layer(x)
        return x


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class TabNet(nn.Module):
    def __init__(self, input_dim,
                 output_dim, n_shared, n_glu, n_steps):
        super(TabNet, self).__init__()
        if n_shared > 0:
            self.shared_layers = nn.ModuleList(
                [GatedLinearUnit(input_dim) for _ in range(n_shared)]
            )
        else:
            self.shared_layers = None

        self.feature_transformers = nn.ModuleList(
            [FeatureTransformer(input_dim, input_dim, self.shared_layers, n_glu) for _ in range(n_steps)]
        )
        self.attentive_transformers = nn.ModuleList(
            [AttentiveTransformer(input_dim, input_dim) for _ in range(n_steps)])

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size, input_dim = x.size()
        output = torch.zeros(batch_size, input_dim).to(x.device)

        for feature_transformer, attentive_transformer in zip(self.feature_transformers, self.attentive_transformers):
            x_transformed = feature_transformer(x)
            mask = attentive_transformer(output)
            x_masked = x_transformed * mask
            output = output + x_masked

        return self.fc(output)

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
