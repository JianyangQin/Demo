import torch
import torch.nn as nn
import pywt


def Wavelet_Intra(x, w='coif1', j=1):

    out = x.detach().cpu().numpy().transpose(0, 3, 2, 1)  # [S,D,N,T]
    coef = pywt.wavedec(out, w, level=j)

    coef_intra = [2 * coef[0]]
    for i in range(len(coef) - 1):
        coef_intra.append(coef[i + 1])

    out = pywt.waverec(coef_intra, w).transpose(0, 3, 2, 1)
    out = torch.from_numpy(out).to(x.device)
    return out


def Wavelet_Inter(x, w='coif1', j=1):

    out = x.detach().cpu().numpy().transpose(0, 3, 2, 1)  # [S,D,N,T]
    coef = pywt.wavedec(out, w, level=j)

    coef_inter = [coef[0]]
    for i in range(len(coef) - 1):
        coef_inter.append(2 * coef[i + 1])

    out = pywt.waverec(coef_inter, w).transpose(0, 3, 2, 1)
    out = torch.from_numpy(out).to(x.device)
    return out


# Encoder for intra flow
class Encoder_Intra(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0., use_batchnorm=False):
        """
        Encoder for intra flow with customizable depth, batch normalization, and dropout.

        Args:
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden layer dimensionality.
            num_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate for regularization.
            use_batchnorm (bool): If True, applies BatchNorm after each layer.
        """
        super().__init__()

        # Define the first layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.initial_activation = nn.ReLU()
        self.initial_dropout = nn.Dropout(dropout_rate)

        # Conditionally add batch normalization
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]) if use_batchnorm else None

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers - 1)])
        # self.tematt = TemporalAttention()

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor (batch_size, num_of_node, sequence_len, input_dim).

        Returns:
            torch.Tensor: Encoded tensor (batch_size, num_of_node, sequence_len, hidden_dim).
        """

        x = Wavelet_Intra(x)

        x = self.initial_layer(x)
        if self.batch_norms:
            x = self.batch_norms[0](x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)

        # Hidden layers forward pass
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[i + 1](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        # print(x.shape)

        return x


# Encoder for inter flow
class Encoder_Inter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0., use_batchnorm=False):
        """
        Encoder for intra flow with customizable depth, batch normalization, and dropout.

        Args:
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden layer dimensionality.
            num_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate for regularization.
            use_batchnorm (bool): If True, applies BatchNorm after each layer.
        """
        super().__init__()

        # Define the first layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.initial_activation = nn.ReLU()
        self.initial_dropout = nn.Dropout(dropout_rate)

        # Conditionally add batch normalization
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]) if use_batchnorm else None

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers - 1)])

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor (batch_size, num_of_node, sequence_len, input_dim).

        Returns:
            torch.Tensor: Encoded tensor (batch_size, num_of_node, sequence_len, hidden_dim).
        """
        # First layer forward pass
        x = Wavelet_Inter(x)
        x = self.initial_layer(x)
        if self.batch_norms:
            x = self.batch_norms[0](x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)

        # Hidden layers forward pass
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[i + 1](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

        return x