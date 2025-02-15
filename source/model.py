import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim):
        super().__init__()
        # self.last_piano_roll = torch.zeros(batch_size, 1, 128)
        # TODO: add intialisation to zeros
        # TODO: 
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True, bias=True)
        self.sigmoid_inter = nn.Sigmoid()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        # self.sigmoid_final = nn.Sigmoid()

    def forward(self, x):    
        # x = torch.cat((x, self.last_piano_roll), dim=2)
        x, (hidden, cell) = self.lstm(x)
        x = self.sigmoid_inter(x)
        x = self.linear(x)
        # x = self.sigmoid_final(x)  # Ensure output is in [0,1] range, since target proll values are normalized to this range
        # save last x for concatenating last midi buffer, intiallized with zeros, save as private var of model
        # self.last_piano_roll = x
        return x