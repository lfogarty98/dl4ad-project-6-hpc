import numpy as np
import torch
from utils import config
from pathlib import Path
from pedalboard.io import AudioFile

import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torchaudio import transforms
import pretty_midi
# from pyprojroot import here
import os

def split_data(data, test_split):
    """
    Splits the data into training and testing sets.
    Assumes that X has shape (1 , num_freq_bins, total_num_frames) and Y has shape (128, total_num_frames).
    """
    split_idx = int(data.shape[-1] * (1 - test_split))
    return np.split(data, [split_idx], axis=-1)

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

def main():
    
    # Workaround for pretty_midi bug
    np.int = int 
    
    # Load the hyperparameters from the params yaml file into a Dictionary
    params = config.Params('params.yaml')

    # Load the parameters from the dictionary into variables
    buffer_size = params['preprocess']['buffer_size']
    n_fft = params['preprocess']['n_fft']
    win_length = params['preprocess']['win_length']
    hop_length = params['preprocess']['hop_length']
    audio_dir = params['preprocess']['audio_dir']
    midi_dir = params['preprocess']['midi_dir']
    test_split = params['preprocess']['test_split']
    
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

    # TODO: normalize specgram values?
    X, Y = prepare_data(audio_files, midi_files, spectrogram_transform)
    print("Data loaded and normalized.")

    # Simple split of data into training and testing sets
    X_training, X_testing = split_data(X, test_split)
    Y_training, Y_testing = split_data(Y, test_split)
    print("Data split into training and testing sets.")

    output_file_path = Path('data/processed/data.pt')
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'X_training': X_training,
        'Y_training': Y_training,
        'X_testing': X_testing,
        'Y_testing': Y_testing
    }, output_file_path)
    print("Preprocessing done and data saved.")

if __name__ == "__main__":
    main()