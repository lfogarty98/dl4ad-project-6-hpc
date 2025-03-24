import torch
import torch.nn as nn
import torchinfo
from utils import logs, config
from pathlib import Path
from model import NeuralNetwork
import pypianoroll as ppr
import matplotlib.pyplot as plt
import os

def regularizer(prediction, threshold=10):
    """
    Regularizer which penalizes frames that have too many active notes by returning a 
    penalty that increases as more notes exceed the given threshold.
    """
    penalty = 0.0  # Initialize penalty
    for frame in prediction:  # Iterate over each frame (Shape: (num_frames, 1, 128))
        active_notes = torch.sum(frame > 0.5).item()  # Count nonzero (active) notes in frame
        if active_notes > threshold:  
            penalty += (active_notes - threshold) ** 2 # Add (linear) penalty for exceeding threshold
    penalty /= prediction.shape[0]  # Average over all frames
    return penalty

def ppr_metrics(prediction, threshold=5):
    """
    Compute metrics related to the predicted piano roll.
    """
    # penalty = torch.tensor(0.0, device='cpu', requires_grad=True)
    penalty = 0.0
    prediction_binary = (prediction > 0.5).float()
    num_pitches = torch.any(prediction_binary.squeeze() != 0, dim=1).sum()
    # prediction_binary = prediction_binary.detach().cpu().numpy().squeeze() # Convert to numpy for ppr libary
    # num_pitches = ppr.n_pitches_used(prediction_binary)
    if num_pitches > threshold:
        penalty += (num_pitches - threshold) ** 2
    return torch.tensor(penalty, requires_grad=True)

def train_epoch(dataloader, model, loss_fn, optimizer, device, writer, epoch, lambda_reg=0.1, max_voices=10):
    """
    Train the model for one epoch using the training dataloader.
    Loss is computed as the sum of the base loss and the regularization penalty.
    Regularization is applied only if lambda_reg > 0.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0 
    # Reset the last_piano_roll state before each training pass
    model.last_piano_roll = torch.zeros(model.batch_size, 1, 128).to(device) 
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        
        # Compute base loss (scaled by 10)
        base_loss = 1 * loss_fn(pred, y)

        # Compute regularization penalty
        reg_loss = lambda_reg * regularizer(torch.sigmoid(pred), threshold=max_voices)
        
        # Compute ppr metrics
        lambda_ppr = 0.01
        ppr_loss = lambda_ppr * ppr_metrics(pred, threshold=6)
        
        # Total loss (base loss + regularization)
        loss = base_loss + reg_loss + ppr_loss 
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("Batch_Loss/train", loss.item(), batch + epoch * len(dataloader))
        writer.add_scalar("Batch_Regularization_Loss/train", reg_loss, batch + epoch * len(dataloader))  # Log reg loss separately
        train_loss += loss.item()
        if batch % 100 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /=  num_batches
    return train_loss
    
def test_epoch(dataloader, model, loss_fn, device, writer):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

def predictions_to_midi(predicted_piano_roll, path):
    predicted_piano_roll = predicted_piano_roll.cpu().numpy().squeeze()
    ppr_object = ppr.Multitrack(tracks=[ppr.StandardTrack(pianoroll=predicted_piano_roll)])
    ppr.write(path, ppr_object)

def generate_predictions(model, device, dataloader, num_eval_batches, start_batch=0):
    """
    Generate predictions (concantenated normalized piano roll matrices) using the model and the testing dataloader.
    Since the training data is effectively one long stream of audio and midi, num_eval_batches specifies the number
    of batches to evaluate, which should be limited for performance reasons.
    
    Returns the predicted piano roll and the target piano roll as matplotlib figures for visualisation.
    
    TODO: Add audio examples
    """
    prediction = torch.zeros(0).to(device)
    target = torch.zeros(0).to(device)
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            if i < start_batch:  # Skip batches before start_batch
                continue
            if i >= start_batch + num_eval_batches:  # Stop after processing num_eval_batches
                break
            
            X = X.to(device)
            y = y.to(device)
            predicted_batch = model(X)
            predicted_batch_binary = (predicted_batch > 0.5).float()  # Binarize the prediction
            prediction = torch.cat((prediction, predicted_batch_binary), 0)
            target = torch.cat((target, y), 0)
    # Create plots
    piano_roll_prediction_plot = plot_piano_roll(prediction, "Predicted Piano Roll")
    piano_roll_target_plot = plot_piano_roll(target, "Target Piano Roll")
    return prediction, piano_roll_prediction_plot, piano_roll_target_plot

def plot_piano_roll(piano_roll, plot_title):
    """
    Plot piano roll using matplotlib, and return the figure object for the Tensorboard logs.
    Converts the piano roll to a numpy array and transposes it to match the expected format for plotting.
    """ 
    piano_roll = piano_roll.cpu().numpy().squeeze()
    piano_roll = piano_roll.T
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(piano_roll, cmap="gray", aspect="auto", origin="lower")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MIDI Note")
    ax.set_title(plot_title)
    return fig

# def plot_piano_roll(piano_roll, plot_title, binarize=False):
#     """
#     Plot a (optionally binarized) piano roll using pypianoroll, and return the figure object for the Tensorboard logs.
#     For visualisation purposes, the piano roll is scaled back to [0, 127] and binarized.
#     """
#     piano_roll_scaled = (piano_roll * 127).astype(int) # Scale back to [0, 127]
#     ppr_object = ppr.Multitrack(tracks=[ppr.StandardTrack(pianoroll=piano_roll_scaled)]) # NOTE: may need to transpose to match pypianoroll format
#     if binarize:
#         ppr_object.binarize() # Binarize the piano roll
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ppr.plot_pianoroll(ax, ppr_object.tracks[0].pianoroll, cmap="Blues")
#     ax.set_title(plot_title)
#     return fig

def reshape_and_batch(X, Y):
    # Add batch dimension to Y
    Y = Y.unsqueeze(0) # Shape: (1, 128, num_frames)
    # Reshape tensors 
    X = X.permute(2, 0, 1)  # Shape: (num_frames, 1, num_freq_bins)
    Y = Y.permute(2, 0, 1)  # Shape: (num_frames, 1, 128)
    return X, Y

def calculate_weight(Y, max_weight=100.0):
    """
    Calculate the positive weight for the BCEWithLogitsLoss loss function.
    The positive weight is the ratio of negative samples to positive samples
    per MIDI note class across all frames.
    """
    Y = Y.squeeze()
    num_positives = Y.sum(dim=0)
    num_negatives = Y.shape[0] - num_positives
    pos_weight = num_negatives / (num_positives + 1e-6)
    pos_weight = torch.clamp(pos_weight, 1.0, max_weight)  # Clamp the weight to a maximum value
    print(f'Positive samples: {num_positives}')
    print(f'Negative samples: {num_negatives}')
    print(f'pos_weight: {pos_weight}')
    return pos_weight

def main():
    # Load the hyperparameters from the params yaml file into a Dictionary
    params = config.Params()

    # Load the parameters from the dictionary into variables
    random_seed = params['general']['random_seed']
    midi_output_dir = params['general']['midi_output_dir']
    epochs = params['train']['epochs']
    batch_size = params['train']['batch_size']
    learning_rate = params['train']['learning_rate']
    device_request = params['train']['device_request']
    num_eval_batches = params['train']['num_eval_batches']
    lambda_reg = params['train']['lambda_reg']
    max_voices = params['train']['max_voices']
    hidden_size = params['model']['hidden_size']
    num_lstm_layers = params['model']['num_lstm_layers']

    # Create a SummaryWriter object to write the tensorboard logs
    tensorboard_path = logs.return_tensorboard_path()
    metrics = {'Epoch_Loss/train': None, 'Epoch_Loss/test': None, 'Batch_Loss/train': None}
    writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics)

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(random_seed)
    
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.prepare_device(device_request)

    # Load preprocessed data from the input file into the training and testing tensors
    input_file_path = Path('data/processed/data.pt')
    data = torch.load(input_file_path)
    X_training = data['X_training']
    Y_training = data['Y_training']
    X_testing = data['X_testing']
    Y_testing = data['Y_testing']
    
    # Create the model
    num_freq_bins = X_training.shape[1] # X has shape (1, num_freq_bins, total_num_frames) (assuming mono audio)
    num_midi_classes = 128
    input_dim = num_freq_bins + num_midi_classes
    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_size,
        num_lstm_layers=num_lstm_layers,
        output_dim=num_midi_classes,
        batch_size=batch_size
    ).to(device)
    
    # Reshape data for the model training
    X_training, Y_training = reshape_and_batch(X_training, Y_training)
    X_testing, Y_testing = reshape_and_batch(X_testing, Y_testing)
    print(f'X_training shape: {X_training.shape}')
    print(f'Y_training shape: {Y_training.shape}')
    
    # Print the model summary
    input_size = (1, 1, num_freq_bins) # shape compliant with the model input
    summary = torchinfo.summary(model, input_size, device=device)
    
    # Add the model graph to the tensorboard logs
    # NOTE: weird behaviour here after adding output feedback to the model
    # sample_inputs = torch.randn(input_size) 
    # writer.add_graph(model, sample_inputs.to(device))

    # Define the loss function and the optimizer
    pos_weight = calculate_weight(Y_training).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)  # Binary cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create the dataloaders
    # NOTE: shuffle disabled to preserve time ordering of frames/batches 
    # NOTE: drop_last enabled to ensure all batches have the same size, so that input-output concatenation in model works
    training_dataset = torch.utils.data.TensorDataset(X_training, Y_training)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    testing_dataset = torch.utils.data.TensorDataset(X_testing, Y_testing)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Training loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_loss_train = train_epoch(
            training_dataloader, 
            model, loss_fn, 
            optimizer, 
            device, 
            writer, 
            epoch=t, 
            lambda_reg=lambda_reg, 
            max_voices=max_voices
        )
        epoch_loss_test = test_epoch(testing_dataloader, model, loss_fn, device, writer)
        writer.add_scalar("Epoch_Loss/train", epoch_loss_train, t)
        writer.add_scalar("Epoch_Loss/test", epoch_loss_test, t)
        piano_roll_training_prediction, piano_roll_training_prediction_plot, piano_roll_training_target_plot = generate_predictions(model, device, training_dataloader, num_eval_batches, start_batch=0)
        writer.add_figure("Piano_Roll/train/prediction", piano_roll_training_prediction_plot, t)
        writer.add_figure("Piano_Roll/train/target", piano_roll_training_target_plot, t)
        piano_roll_test_prediction, piano_roll_test_prediction_plot, piano_roll_test_target_plot = generate_predictions(model, device, testing_dataloader, num_eval_batches, start_batch=0)
        writer.add_figure("Piano_Roll/test/prediction", piano_roll_test_prediction_plot, t)
        writer.add_figure("Piano_Roll/test/target", piano_roll_test_target_plot, t)
        # TODO: add MIDI output
        # if t % 50 == 0: 
        #     midi_output_path = os.path.join(midi_output_dir, f'output_training_{t}')
        #     predictions_to_midi(piano_roll_training_prediction, midi_output_path)
        #     midi_output_path = os.path.join(midi_output_dir, f'output_test_{t}')
        #     predictions_to_midi(piano_roll_test_prediction, midi_output_path)
        writer.step()  

    writer.close()

    # Save the model checkpoint
    output_file_path = Path('models/checkpoints/model.pth')
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_file_path)
    print("Saved PyTorch Model State to model.pth")

    print("Done with the training stage!")

if __name__ == "__main__":
    main()
