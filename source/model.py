import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size # initialize last_piano_roll with zeros
        self.last_piano_roll = torch.zeros(batch_size, 1, 128).to('cuda')
        # NOTE: input_dim is num_freq_bins + num_midi_classes
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_lstm_layers, 
            batch_first=True, 
            bias=True
        )
        self.sigmoid_inter = nn.Sigmoid()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        self.last_piano_roll = self.last_piano_roll.detach() # NOTE: avoids backward problem, not quite sure how this works
        x = torch.cat((x, self.last_piano_roll[:x.shape[0], :, :]), dim=-1)
        x, (hidden, cell) = self.lstm(x)
        x = self.sigmoid_inter(x)
        x = self.linear(x)
        self.last_piano_roll = x
        return x