# import all important libraries
import torch
import time
import json
import argparse
import logging
import pandas as pd
from model import Model
# Device configuration
device = torch.device("cpu")


def read(name):
    """
            Parameters
            ----------
            name : csv file
                The name of the test dataset file
            ----------
            Description
            ----------
            This function read sequences from the csv file,
            and makes it ready in the desired format for prediction.

            """
    lists = []
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            lists.append(tuple(ln))
    try:
        if lists:
            return lists
    except Exception:
        logging.error("Could not read data")


def execute():

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', type=int)
    parser.add_argument('-hidden_size', type=int)
    parser.add_argument('-window_size', type=int)
    parser.add_argument('-num_candidates', type=int)#100

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info('model_path: {}'.format(model_path))

    # Loading Test file that contains normal logs
    try:
        test_normal_loader = read("Test_Out.csv")
        logging.debug("Testing data read")
    except Exception:
        logging.critical("Could not read Test data")
    test_normal_loader = read("Test_Out.csv")
    # Test the saved_model
    start_time = time.time()

    with torch.no_grad():

        p = {}
        k = 0

        m = {}
        m1 = 0
        for line in test_normal_loader:

            m1 = m1+1
            m[m1] = []
            for i in range(len(line) - window_size):

                k = k + 1
                p[k] = []

                seq = line[i:i + window_size]
                label = line[i + window_size]
                j = label+1    # the next key
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)  # the next predicted key
                output = model(seq)  # the scores generated for each class to be the next key element
                predicted = torch.argsort(output, 1)[0][-num_candidates:]

                # if next key is not in those predicted ones

                if label not in predicted:
                    p[k].append(j)

                elif label in predicted:
                    p[k] = []
                m[m1].append(p[k])

        # print(value)
        # print(p)
        data_df = pd.DataFrame(list(m.items()), columns=['PID', 'Pred_Sequence'])
        data_df.to_csv(r"data\output.csv", index=None)
        # print(m)


if __name__ == '__main__':

    f = open('configs/params.json', )
    data = json.load(f)
    num_classes = int(data["model_configs_test"]["num_classes"])
    input_size = int(data["model_configs_test"]["input_size"])
    model_path = data["model_configs_test"]["model_path"]
    num_layers = int(data["model_configs_test"]["num_layers"])
    hidden_size = int(data["model_configs_test"]["hidden_size"])
    window_size = int(data["model_configs_test"]["window_size"])
    num_candidates = int(data["model_configs_test"]["num_candidates"])

    try:
        execute()
        f.close()
    except Exception:
        logging.error("Could not start the process")
