import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim, batch_size):
        super().__init__()
        # add intialisation to zeros
        self.last_piano_roll = torch.zeros(batch_size, 1, 128)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True, bias=True)
        self.sigmoid_inter = nn.Sigmoid()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):    
        print(f'Input shape: {x.shape}')
        self.last_piano_roll = self.last_piano_roll.detach()
        x = torch.cat((x, self.last_piano_roll), dim=-1)
        x, (hidden, cell) = self.lstm(x)
        x = self.sigmoid_inter(x)
        x = self.linear(x)
        # save last x for concatenating last midi buffer, intiallized with zeros, save as private var of model
        self.last_piano_roll = x
        return x