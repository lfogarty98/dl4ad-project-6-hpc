import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torchaudio import transforms
import pretty_midi
# from pyprojroot import here
import os

from utils import config

# data_dir = here('data')

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def audio_to_spectrogram(audio_file, spectrogram_transform):
    waveform, sample_rate = torchaudio.load(audio_file)
    # Check if the audio is stereo (multi-channel) and convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    print(f'Audio filepath: {audio_file})')
    print(f'Duration of waveform: {waveform.shape[1] / sample_rate} seconds')
    specgram = spectrogram_transform(waveform)
    return specgram, sample_rate

def midi_to_piano_roll(midi_file, fs):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    print(f'MIDI filepath: {midi_file})')
    print(f"Duration of MIDI file: {midi_data.get_end_time()} seconds ")
    print(f'Number of ticks: {midi_data.time_to_tick(midi_data.get_end_time())}')
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return piano_roll

def prepare_data(audio_files, midi_files, spectrogram_transform):
    X, Y = [], []
    for audio_file, midi_file in zip(audio_files, midi_files):
        
        # Compute spectrogram
        specgram, sr = audio_to_spectrogram(audio_file, spectrogram_transform)
        
        # Compute sampling frequency of the columns for the piano roll, based on number of frames in specgram
        num_frames = specgram.shape[-1] # Number of frames in the spectrogram bzw. columns in specgram matrix
        T = num_frames * spectrogram_transform.hop_length / sr # Duration of audio/midi in seconds
        fs = num_frames / T # each column is spaced apart by 1./fs seconds.
        
        # Compute piano roll
        piano_roll = midi_to_piano_roll(midi_file, fs=fs)
        
        # Trim or pad piano roll to match the number of frames in the spectrogram, assuming mismatch is due to rounding errors
        if piano_roll.shape[-1] > num_frames:
            piano_roll = piano_roll[:, :num_frames]  # Trim excess columns in piano roll
        elif piano_roll.shape[-1] < num_frames:
            padding = num_frames - piano_roll.shape[-1]
            piano_roll = np.pad(piano_roll, ((0, 0), (0, padding)), mode='constant')  # Pad piano roll with zeros
            
        # Normalization of piano roll values to [0, 1]
        piano_roll_normalized = piano_roll / np.max(piano_roll) # since min of piano roll is 0, we can just divide by max to normalize
        
        X.append(specgram)
        Y.append(piano_roll_normalized)

    # Concatenate all spectrograms and piano rolls
    X_flat = torch.cat(X, dim=-1)
    Y_flat = torch.cat([torch.tensor(y, dtype=torch.float32) for y in Y], dim=-1)
    return X_flat, Y_flat

def predictions_to_midi():
    # TODO: Implement this function
    pass

def build_rnn_model(input_dim, hidden_size, num_lstm_layers, output_dim):
    rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_lstm_layers, batch_first=True)
    fc = nn.Linear(hidden_size, output_dim)
    model = nn.Sequential(rnn, fc)
    return model

def train_model(model, X, Y, epochs, batch_size=16, learning_rate=1e-3):
    # Ensure the model is in training mode
    model.train()

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # MSE for piano roll values in [0,1]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Add batch dimension to Y
    Y = Y.unsqueeze(0) # Shape: (1, 128, num_frames)
    
    # Reshape tensors 
    X = X.permute(2, 0, 1)  # Shape: (num_frames, 1, num_freq_bins)
    Y = Y.permute(2, 0, 1)  # Shape: (num_frames, 1, 128)
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0

        for i in range(0, len(X), batch_size):
            # Create batches
            X_batch = X[i:i + batch_size]  # Shape: (batch_size, 1, num_freq_bins)
            Y_batch = Y[i:i + batch_size] # Shape: (batch_size, 128)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model[0](X_batch)  # LSTM forward pass
            predictions = model[1](outputs)  # Fully connected layer forward pass

            # Compute loss
            loss = criterion(predictions, Y_batch)

            # Log loss to TensorBoard
            writer.add_scalar("Loss/train", loss, epoch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X):.4f}")

    print("Training completed.")
    
    # Flush and close the TensorBoard writer
    writer.flush()
    writer.close()
    
    return

def main():
    # Workaround for pretty_midi bug
    np.int = int 

    params = config.Params()

    buffer_size = params['general']['buffer_size']
    n_fft = params['general']['n_fft']
    win_length = params['general']['win_length']
    hop_length = params['general']['hop_length']
    hidden_size = params['general']['hidden_size']
    num_lstm_layers = params['general']['num_lstm_layers']
    epochs = params['general']['epochs']
    batch_size = params['general']['batch_size']
    learning_rate = params['general']['learning_rate']
    audio_dir = params['general']['audio_dir']
    midi_dir = params['general']['midi_dir']

    # Create the Spectrogram transform
    spectrogram_transform = transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0  # Power of 2.0 gives the magnitude squared (power spectrogram)
    )
    
    # Load raw data
    audio_files = [os.path.join(audio_dir, filename) for filename in os.listdir(audio_dir)]
    midi_files = [os.path.join(midi_dir, filename) for filename in os.listdir(midi_dir)]
    
    audio_metadata = torchaudio.info(audio_files[0])
    print(audio_metadata)
    
    # Prepare data
    X, Y = prepare_data(audio_files, midi_files, spectrogram_transform)
    print(X.shape) # X has shape (1, num_freq_bins, total_num_frames) (assuming mono audio)
    print(Y.shape) # Y has shape (128, total_num_frames)
    
    # Build and train model
    num_freq_bins = n_fft // 2 + 1
    num_midi_classes = 128
    model = build_rnn_model(input_dim=num_freq_bins, hidden_size=hidden_size, num_lstm_layers=num_lstm_layers, output_dim=num_midi_classes)
    train_model(model, X, Y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # Create checkpoint directory
    ckt_dir = "models/checkpoints"
    os.makedirs(ckt_dir, exist_ok=True)
    breakpoint()
    
    # Save model
    torch.save(model, os.path.join(ckt_dir, "rnn_model_full.pth"))

if __name__ == "__main__":
    main()


# from torch import nn

# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, bias=True)
#         self.linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

#     def forward(self, x):
#         x, (hidden, cell) = self.lstm(x)
#         x = self.linear(x)
#         return x