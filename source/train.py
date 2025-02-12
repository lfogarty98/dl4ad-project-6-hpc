import torch
import torch.nn as nn
import torchinfo
from utils import logs, config
from pathlib import Path
from model import NeuralNetwork
import pypianoroll as ppr
import matplotlib.pyplot as plt

def train_epoch(dataloader, model, loss_fn, optimizer, device, writer, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0 
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = 10 * loss_fn(pred, y) # Multiply by 10 to scale the loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("Batch_Loss/train", loss.item(), batch + epoch * len(dataloader))
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

def predictions_to_midi(predictions):
    pass

def generate_predictions(model, device, dataloader, num_eval_batches):
    """
    Generate predictions (concantenated normalized piano roll matrices) using the model and the testing dataloader.
    Since the training data is effectively one long stream of audio and midi, num_eval_batches specifies the number
    of batches to evaluate, which should be limited for performance reasons.
    
    Returns the predicted piano roll and the target piano roll as matplotlib figures for visualisation.
    """
    prediction = torch.zeros(0).to(device)
    target = torch.zeros(0).to(device)
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            if i >= num_eval_batches: # Limit the number of batches to evaluate
                break
            
            X = X.to(device)
            y = y.to(device)
            predicted_batch = model(X)
            
            prediction = torch.cat((prediction, predicted_batch), 0)
            target = torch.cat((target, y), 0)
    # Convert prediction and target from normalized proll to plots
    prediction = prediction.cpu().numpy().squeeze() 
    target = target.cpu().numpy().squeeze()
    piano_roll_prediction_plot = plot_binarized_piano_roll(prediction, "Predicted Piano Roll")
    piano_roll_target_plot = plot_binarized_piano_roll(target, "Target Piano Roll")
    return piano_roll_prediction_plot, piano_roll_target_plot

def plot_binarized_piano_roll(piano_roll, plot_title):
    piano_roll_scaled = (piano_roll * 127).astype(int) # Scale back to [0, 127]
    ppr_object = ppr.Multitrack(tracks=[ppr.StandardTrack(pianoroll=piano_roll_scaled.T)]) # Transpose to match pypianoroll format
    ppr_object.binarize()
    fig, ax = plt.subplots(figsize=(12, 6))
    ppr.plot_pianoroll(ax, ppr_object.tracks[0].pianoroll, cmap="Blues")
    ax.set_title(plot_title)
    return fig

def reshape_and_batch(X, Y):
    # Add batch dimension to Y
    Y = Y.unsqueeze(0) # Shape: (1, 128, num_frames)
    # Reshape tensors 
    X = X.permute(2, 0, 1)  # Shape: (num_frames, 1, num_freq_bins)
    Y = Y.permute(2, 0, 1)  # Shape: (num_frames, 1, 128)
    return X, Y

def main():
    # Load the hyperparameters from the params yaml file into a Dictionary
    params = config.Params()

    # Load the parameters from the dictionary into variables
    random_seed = params['general']['random_seed']
    epochs = params['train']['epochs']
    batch_size = params['train']['batch_size']
    learning_rate = params['train']['learning_rate']
    device_request = params['train']['device_request']
    num_eval_batches = params['train']['num_eval_batches']
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
    model = NeuralNetwork(input_dim=num_freq_bins, hidden_dim=hidden_size, num_lstm_layers=num_lstm_layers, output_dim=num_midi_classes)
    
    # Reshape data for the model training
    X_training, Y_training = reshape_and_batch(X_training, Y_training)
    X_testing, Y_testing = reshape_and_batch(X_testing, Y_testing)

    # Print the model summary
    input_size = (batch_size, 1, num_freq_bins) # shape compliant with the model input
    summary = torchinfo.summary(model, input_size, device=device)

    # Add the model graph to the tensorboard logs
    sample_inputs = torch.randn(input_size) 
    writer.add_graph(model, sample_inputs.to(device))

    # Define the loss function and the optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create the dataloaders
    training_dataset = torch.utils.data.TensorDataset(X_training, Y_training)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False) # NOTE: does it make sense to shuffle?
    testing_dataset = torch.utils.data.TensorDataset(X_testing, Y_testing)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_loss_train = train_epoch(training_dataloader, model, loss_fn, optimizer, device, writer, epoch=t)
        epoch_loss_test = test_epoch(testing_dataloader, model, loss_fn, device, writer)
        writer.add_scalar("Epoch_Loss/train", epoch_loss_train, t)
        writer.add_scalar("Epoch_Loss/test", epoch_loss_test, t)
        # TODO: add audio examples
        piano_roll_prediction_plot, piano_roll_target_plot = generate_predictions(model, device, testing_dataloader, num_eval_batches)
        writer.add_figure("Piano_Roll/prediction", piano_roll_prediction_plot, t)
        writer.add_figure("Piano_Roll/target", piano_roll_target_plot, t)
        # epoch_audio_prediction, epoch_audio_target  = generate_audio_examples(model, device, testing_dataloader, num_eval_batches)
        # writer.add_audio("Audio/prediction", epoch_audio_prediction, t, sample_rate=44100)
        # writer.add_audio("Audio/target", epoch_audio_target, t, sample_rate=44100)        
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
