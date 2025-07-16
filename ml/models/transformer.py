from types import SimpleNamespace
from typing import List

import torch

from ml.utils.train_utils import fit, predict


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, in_size, num_heads: int = 10):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = torch.nn.MultiheadAttention(in_size, num_heads)

    def forward(self, x):
        attention_out, _ = self.attention(x, x, x)
        return attention_out


class TabularTransformer(torch.nn.Module):
    def __init__(self, in_size, out_size,
                 hidden_dims,
                 num_heads, num_layers, dropout):
        super(TabularTransformer, self).__init__()

        self.input_layer = torch.nn.Linear(in_size, hidden_dims[0])
        self.norm_input = torch.nn.LayerNorm(hidden_dims[0])

        self.layers = torch.nn.ModuleList()
        self.attentions = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.attentions.append(MultiHeadAttention(hidden_dims[i+1], num_heads))
            self.norms.append(torch.nn.LayerNorm(hidden_dims[i+1]))

        self.dropout = torch.nn.Dropout(dropout)
        self.output = torch.nn.Linear(hidden_dims[-1], out_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.norm_input(x)
        x = torch.nn.functional.relu(x)

        for layer, attention, norm in zip(self.layers, self.attentions, self.norms):
            x = layer(x)
            x = torch.nn.functional.relu(x)
            x = x.unsqueeze(1)  # Add sequence length dimension
            x = x.permute(1, 0, 2)  # Permute dimensions for multi-head attention
            x = attention(x)
            x = x.permute(1, 0, 2)  # Revert dimensions to original
            x = x.squeeze(1)  # Remove sequence length dimension
            x = norm(x)

        x = self.dropout(x)
        x = self.output(x)

        return x

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
