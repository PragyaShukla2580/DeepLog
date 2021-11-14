# import all important libraries
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import numpy as np
import logging
from model import Model
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(name):

    """
            Parameters
            ----------
            name : csv file
                The name of the train dataset file


            Description
            ----------
            This function read sequences from the csv file,
            split the sequences into input and output format
            and makes it ready for the saved_model to train.

            """
    num_sessions = 0
    inputs = []
    outputs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                # inputs list takes the sequences of length window_size
                inputs.append(line[i:i + window_size])
                # outputs list takes the next key
                outputs.append(line[i + window_size])

    # logging.info('Number of sessions({}): {}'.format(name, num_sessions))
    # logging.info('Number of seqs({}): {}'.format(name, len(inputs)))

    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    try:
        if dataset:
            logging.debug("File generated from data to go for training")
            return dataset

    except Exception:
        logging.error("Error in generating file")
        return


def execute():

    # log states the name of saved_model in terms of batch_size and epoch
    log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))

    # setting default num_layers, hidden_size, window_size
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', type=int)
    parser.add_argument('-hidden_size', type=int)
    parser.add_argument('-window_size', type=int)

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    try:
        seq_dataset = generate('Train.csv')
    except Exception:
        logging.critical("Training file not found")
        return

    # dataloader
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    writer = SummaryWriter(log_dir='log/' + log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the saved_model
    start_time = time.time()

    # initialize the number of passes and iterations to be 0
    epochs_no_improve = 0
    iterations = 0
    early_stop = False

    # set min_val_loss as infinity
    min_val_loss = np.Inf
    logging.info("Training starting")
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):

            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Check for Early Stopping
            if train_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = train_loss
            else:
                epochs_no_improve += 1
            iterations += 1

            if epoch > 10 and epochs_no_improve == n_epochs_stop:
                logging.info("Early Stopping")
                early_stop = True
                break
            else:
                continue

        # Early Stopping
        if early_stop:
            logging.info("Early Stopped")
            break

        # At the end of each epoch, print the loss
        writer.add_graph(model, seq)
        logging.info('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step)) #added to keep track in local system
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time

    # Print the total time taken to train
    logging.info('elapsed_time: {:.3f}s'.format(elapsed_time))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Save the saved_model
    try:
        torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
        writer.close()
    except Exception:
        logging.error("Model not saved")


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
    try:
        execute()
        f.close()
    except Exception:
        logging.error("Training not started")