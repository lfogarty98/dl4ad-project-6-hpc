import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lstm_layers, output_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size # initialize last_piano_roll with zeros
        self.last_piano_roll = torch.zeros(batch_size, 1, 128)
        # NOTE: input_dim is num_freq_bins + num_midi_classes
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_lstm_layers, 
            batch_first=False, 
            bias=True
        )
        self.sigmoid_inter = nn.Sigmoid()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
        torch.nn.init.zeros_(self.linear.bias) # Zero initialization
        self.hidden = torch.zeros(num_lstm_layers, 1, hidden_dim)
        self.cell = torch.zeros(num_lstm_layers, 1, hidden_dim)


    def forward(self, x):
        # self.last_piano_roll = self.last_piano_roll.detach() # NOTE: avoids backward problem, not quite sure how this works
        # x = torch.cat((x, self.last_piano_roll[:x.shape[0], :, :]), dim=-1)
        x, (hidden, cell) = self.lstm(x, (self.hidden, self.cell))
        self.hidden = hidden
        self.cell = cell
        # x = self.sigmoid_inter(x)
        # x = self.linear(x)
        self.last_piano_roll = x
        return x
    
    def detach_hidden(self):
        self.hidden = self.hidden.detach()
        self.cell = self.cell.detach()
        

"""
This model is for debugging purposes.
"""
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_lstm_layers=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers)
        self.hidden = torch.zeros(self.num_lstm_layers, 1, hidden_dim) # NOTE: docs say hidden should be of shape (D, N, H)
        self.cell = torch.zeros(self.num_lstm_layers, 1, hidden_dim)
        
        for layer in range(self.num_lstm_layers):
            torch.nn.init.xavier_uniform_(getattr(self.lstm, f'weight_ih_l{layer}'))
            torch.nn.init.xavier_uniform_(getattr(self.lstm, f'weight_hh_l{layer}'))
            torch.nn.init.zeros_(getattr(self.lstm, f'bias_ih_l{layer}'))
            torch.nn.init.zeros_(getattr(self.lstm, f'bias_hh_l{layer}'))
        
        self.relu = nn.ReLU()
        
        self.linear_stack = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), # for connecting to LSTM
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        x = self.relu(x)
        x = self.linear_stack(x)
        return x
    
    def detach_hidden(self):
        self.hidden = self.hidden.detach()
        self.cell = self.cell.detach()
        
    def reset_hidden(self):
        self.hidden = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim)
        self.cell = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim)