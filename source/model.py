import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True, bias=True)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x, (hidden, cell) = self.lstm(x)
        x = self.linear(x)
        x = torch.sigmoid(x) * 127  # Ensures output in [0, 127]
        return x