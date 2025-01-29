import numpy as np
import torch
from utils import config
from pathlib import Path
from pedalboard.io import AudioFile

import torchaudio
import pretty_midi

def normalize(data):
    data_norm = max(max(data), abs(min(data)))
    return data / data_norm

def load_and_process_audio(file_path):
    with AudioFile(file_path) as f:
        data = f.read(f.frames).flatten().astype(np.float32)
    return normalize(data)

def split_data(data, test_split):
    split_idx = int(len(data) * (1 - test_split))
    return np.split(data, [split_idx])

def create_ordered_data(data, input_size):
    indices = np.arange(input_size) + np.arange(len(data)-input_size+1)[:, np.newaxis]
    indices = torch.from_numpy(indices)
    data = torch.from_numpy(data)
    ordered_data = torch.zeros_like(indices, dtype=torch.float32)
    for i, idx in enumerate(indices):
        ordered_data[i] = torch.gather(data, 0, idx)
    return ordered_data.unsqueeze(1)

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

def prepare_data(audio_files, midi_files):
    X, Y = [], []
    for audio_file, midi_file in zip(audio_files, midi_files):
        
        # Compute spectrogram
        specgram, sr = audio_to_spectrogram(audio_file, spectrogram_transform)
        
        # Compute sampling frequency of the columns for the piano roll, based on number of frames in specgram
        num_frames = specgram.shape[-1] # Number of frames in the spectrogram bzw. columns in specgram matrix
        T = num_frames * hop_length / sr # Duration of audio/midi in seconds
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
    # Load the hyperparameters from the params yaml file into a Dictionary
    params = config.Params('params.yaml')

    # Load the parameters from the dictionary into variables
    input_size = params['general']['input_size']
    input_file = params['preprocess']['input_file']
    target_file = params['preprocess']['target_file']
    test_split = params['preprocess']['test_split']

    X_all = load_and_process_audio(input_file)
    y_all = load_and_process_audio(target_file)
    print("Data loaded and normalized.")

    X_training, X_testing = split_data(X_all, test_split)
    y_training, y_testing = split_data(y_all, test_split)
    print("Data split into training and testing sets.")

    X_ordered_training = create_ordered_data(X_training, input_size)
    X_ordered_testing = create_ordered_data(X_testing, input_size)
    print("Input data ordered.")

    y_ordered_training = torch.from_numpy(y_training[input_size-1:]).unsqueeze(1)
    y_ordered_testing = torch.from_numpy(y_testing[input_size-1:]).unsqueeze(1)
    print("Target data ordered.")

    output_file_path = Path('data/processed/data.pt')
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'X_ordered_training': X_ordered_training,
        'y_ordered_training': y_ordered_training,
        'X_ordered_testing': X_ordered_testing,
        'y_ordered_testing': y_ordered_testing
    }, output_file_path)
    print("Preprocessing done and data saved.")

if __name__ == "__main__":
    main()