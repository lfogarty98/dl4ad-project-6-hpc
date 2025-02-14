import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True, bias=True)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, (hidden, cell) = self.lstm(x)
        x = self.linear(x)
        x = self.sigmoid(x)  # Ensure output is in [0,1] range, since target proll values are normalized to this range
        return x