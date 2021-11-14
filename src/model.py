import time
import torch
import torch.nn as nn
import logging
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    """
        A class used to create the LSTM saved_model.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        # hidden size
        self.hidden_size = hidden_size
        # number of hidden layers
        self.num_layers = num_layers
        # Building the LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_size, num_keys)
        logging.info("Model loaded")

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        try:
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
        except Exception:
            logging.error("Error in saved_model configurations")
            return


if __name__ == '__main__':
    f = open('configs/params.json', )
    data = json.load(f)
    hidden_size = int(data["model_configs_train"]["hidden_size"])
    num_layers = int(data["model_configs_train"]["num_layers"])
    input_size = int(data["model_configs_train"]["input_size"])
    num_classes = int(data["model_configs_train"]["num_classes"])
    num_epochs = int(data["model_configs_train"]["num_epochs"])
    batch_size = int(data["model_configs_train"]["batch_size"])
    window_size = int(data["model_configs_train"]["window_size"])
    n_epochs_stop = int(data["model_configs_train"]["n_epochs_stop"])
    model_dir = data["model_configs_train"]["model_dir"]

    Model(input_size,hidden_size,num_layers,num_classes)
