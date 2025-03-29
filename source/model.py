import torch
from torch import nn

"""
This model is a simple feedforward neural network, consisting solely of 
linear layers with ReLU activation functions.
"""
class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.linear_stack = nn.Sequential(*layers)
        
        # Initialize weights
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.linear_stack(x)
        return x

"""
This model prepends an LSTM layer to a stack of linear layers.
"""
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim_lstm, hidden_dim_linear, output_dim, num_linear_layers=3, num_lstm_layers=1, device='cuda'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim_lstm = hidden_dim_lstm
        self.hidden_dim_linear = hidden_dim_linear
        self.num_lstm_layers = num_lstm_layers
        self.num_linear_layers = num_linear_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim_lstm, num_layers=self.num_lstm_layers)
        self.hidden = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim_lstm, device=self.device) # NOTE: docs say hidden should be of shape (D, N, H)
        self.cell = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim_lstm, device=self.device)
        
        for layer in range(self.num_lstm_layers):
            torch.nn.init.xavier_uniform_(getattr(self.lstm, f'weight_ih_l{layer}'))
            torch.nn.init.xavier_uniform_(getattr(self.lstm, f'weight_hh_l{layer}'))
            torch.nn.init.zeros_(getattr(self.lstm, f'bias_ih_l{layer}'))
            torch.nn.init.zeros_(getattr(self.lstm, f'bias_hh_l{layer}'))
        
        self.relu = nn.ReLU()
        
        linear_layers = []
        linear_layers.append(nn.Linear(self.hidden_dim_lstm, self.hidden_dim_linear))
        linear_layers.append(nn.ReLU())
        for _ in range(self.num_linear_layers - 1):
            linear_layers.append(nn.Linear(self.hidden_dim_linear, self.hidden_dim_linear))
            linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(self.hidden_dim_linear, output_dim))
        
        self.linear_stack = nn.Sequential(*linear_layers)
        
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
        self.hidden = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim_lstm, device=self.device)
        self.cell = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim_lstm, device=self.device)


"""
This model is the same as LSTMNetwork, but includes output feedback.
"""
class FeedbackLSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim_lstm, hidden_dim_linear, output_dim, num_linear_layers, num_lstm_layers, batch_size, device='cuda'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim_lstm = hidden_dim_lstm
        self.hidden_dim_linear = hidden_dim_linear
        self.num_lstm_layers = num_lstm_layers
        self.num_linear_layers = num_linear_layers
        self.device = device
        self.batch_size = batch_size
        self.last_piano_roll = torch.zeros(self.batch_size, 1, 128, device=self.device) # initialize last_piano_roll with zeros
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim_lstm, num_layers=self.num_lstm_layers)
        self.hidden = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim_lstm, device=self.device) # NOTE: docs say hidden should be of shape (D, N, H)
        self.cell = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim_lstm, device=self.device)
        
        for layer in range(self.num_lstm_layers):
            torch.nn.init.xavier_uniform_(getattr(self.lstm, f'weight_ih_l{layer}'))
            torch.nn.init.xavier_uniform_(getattr(self.lstm, f'weight_hh_l{layer}'))
            torch.nn.init.zeros_(getattr(self.lstm, f'bias_ih_l{layer}'))
            torch.nn.init.zeros_(getattr(self.lstm, f'bias_hh_l{layer}'))
        
        self.relu = nn.ReLU()
        
        linear_layers = []
        linear_layers.append(nn.Linear(self.hidden_dim_lstm, self.hidden_dim_linear))
        linear_layers.append(nn.ReLU())
        for _ in range(self.num_linear_layers - 1):
            linear_layers.append(nn.Linear(self.hidden_dim_linear, self.hidden_dim_linear))
            linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(self.hidden_dim_linear, output_dim))
        
        self.linear_stack = nn.Sequential(*linear_layers)
        
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.cat((x, self.last_piano_roll[:x.shape[0], :, :]), dim=-1) 
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        x = self.relu(x)
        x = self.linear_stack(x)
        self.last_piano_roll = x
        return x
    
    def detach_hidden(self):
        self.hidden = self.hidden.detach()
        self.cell = self.cell.detach()
        
    def detach_piano_roll(self):
        self.last_piano_roll = self.last_piano_roll.detach()
            
    def reset_hidden(self):
        self.hidden = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim_lstm, device=self.device)
        self.cell = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim_lstm, device=self.device)

    def reset_piano_roll(self):
        self.last_piano_roll = torch.zeros(self.batch_size, 1, 128, device=self.device)

"""
Original model we began with, which consists of an LSTM layer followed by a linear layer.
Includes a feedback loop that feeds the output of the linear layer back into the LSTM layer,
such that the last predicted piano roll is used as input for the next prediction.
Currently not functional, and is missing some key components.
"""
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