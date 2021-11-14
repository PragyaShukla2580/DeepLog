# import all important libraries
import torch
import json
import time
import argparse
from model import Model

# Device configuration
device = torch.device("cpu")

# IMPLEMENTATION FOR CALCULATING F1-SCORE


def generate(name):
    """
            Parameters
            ----------
            name : csv file
                The name of the test dataset file


            Description
            ----------
            This function read sequences from the csv file,
            and makes it ready in the desired format for prediction.

            """
    lists =[]
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n-1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            lists.append(tuple(ln))
    # print('Number of sessions({}): {}'.format(name, len(lists)))
    return lists


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

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', type=int) #2
    parser.add_argument('-hidden_size', type=int)
    parser.add_argument('-window_size', type=int)  #10
    parser.add_argument('-num_candidates', type=int) #100

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))

    # Loading Test file that contains normal logs
    test_normal_loader = generate("final_test_normal.csv")

    # Loading Test file that contains abnormal logs
    test_abnormal_loader = generate('final_test_ab.csv')

    TP = 0  # True Positive
    FP = 0  # False Positive

    # Test the saved_model
    start_time = time.time()
    with torch.no_grad():
        for line in test_normal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]          # the sequences which are used to predict next key
                # Example: (58, 45, 58, 143, 57, 145, 59, 6, 83, 50)

                label = line[i + window_size]          # the next key
                # Example: 115

                # sequences are passed to saved_model to predict next key
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                # They become:
                # tensor([[[58.],
                #          [45.],
                #          [58.],
                #          [143.],
                #          [57.],
                #          [145.],
                #          [59.],
                #          [6.],
                #          [83.],
                #          [50.]]])
                label = torch.tensor(label).view(-1).to(device)  # the next predicted key
                # Example: tensor([115])
                output = model(seq)  # the scores generated for each class to be the next key element
                # tensor([[-3.0783e-02, -8.9943e-01, 8.1564e-01, -1.0233e+00, -1.9987e+00,
                #          4.1297e+00, 3.6583e+00, 2.7405e+00, -2.6024e+00, -1.3110e+00,
                #          1.4624e+00, -1.1888e+00, 2.3171e+00, -1.0023e+00, 1.5924e+00,
                #          -9.7280e-01, -1.0281e+00, 5.6696e-01, -1.7781e+00, 4.9569e-01,
                #          1.6300e+00, -2.2752e+00, -2.6482e+00, -2.3190e+00, -9.3338e-01,
                #          -2.0691e+00, -4.5351e-01, 2.6570e+00, -2.5324e-01, -2.6367e-01,
                #          -1.7887e+00, -2.1539e+00, -7.0972e-01, 1.9731e+00, 1.2003e+00,
                #          2.2654e+00, -2.4992e+00, -2.2822e+00, -7.1784e-01, -2.5612e+00,
                #          3.7174e+00, 6.4367e-01, 5.8338e-01, -1.8219e+00, -2.4418e+00,
                #          7.2676e-01, -2.0197e+00, -2.2010e+00, 6.7197e-01, 9.6824e-01,
                #          4.4793e+00, -1.2804e+00, 3.0436e+00, -1.0849e+00, -2.2111e+00,
                #          -2.9887e-01, -8.0357e-01, 1.0036e+00, 5.3157e-01, 2.5761e+00,
                #          7.0003e-01, 9.5440e-01, -1.4410e-01, -1.8530e+00, -1.0610e+00,
                #          2.9231e+00, -6.0022e-01, -1.9135e+00, -8.4950e-01, 1.8913e+00,
                #          -2.2245e+00, -2.4504e+00, 1.9562e+00, -8.9958e-01, 7.7223e-01,
                #          -1.0737e+00, -1.7921e+00, -6.9580e-01, -4.5355e-01, -2.7158e-01,
                #          -6.2391e-01, -7.0932e-01, -1.2717e+00, 3.6388e+00, 5.5729e+00,
                #          -2.3863e+00, -1.2682e-01, -2.6480e+00, -2.5617e+00, -1.1845e+00,
                #          -3.5601e-03, -9.1629e-01, 3.4294e-01, -1.3887e+00, -2.4663e+00,
                #          -1.3307e+00, -1.8597e+00, 1.0854e+00, 7.8676e-01, 2.1777e+00,
                #          -2.6854e+00, 1.2707e+00, 2.2022e-01, -6.3909e-01, 1.2691e+00,
                #          -2.5731e+00, -1.3620e-01, -1.6283e+00, 3.1646e-01, -1.2815e+00,
                #          -2.2624e+00, 5.6995e-01, 2.3623e-01, -2.3656e+00, -2.9465e+00,
                #          6.5139e+00, -2.2631e+00, -2.1908e+00, -1.7394e-01, -2.5614e+00,
                #          -1.1891e+00, 6.9230e-01, 2.4393e-01, -4.0905e-01, 4.7159e-01,
                #          -3.5232e+00, -1.8300e+00, -2.1943e+00, 2.6509e-01, -1.0659e+00,
                #          1.8600e+00, -2.2147e+00, -8.9756e-01, -6.3537e-01, -2.7713e+00,
                #          1.5375e+00, -1.7162e+00, 1.0250e+00, 1.0219e+00, -2.2607e+00,
                #          -1.8459e+00, -1.6357e+00, -2.8957e-01, 1.2824e+00, -2.3566e+00,
                #          1.6755e+00, 3.3733e-02, -1.5107e+00, 2.1677e+00, -2.5073e+00,
                #          -1.4277e+00, 1.2255e+00, -1.8551e+00]])

                # sorting all the scores in descending order and taking the first num_candidates as predicted
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                # Example:  tensor([40, 5, 50, 84, 115])

                # if next key is not in those predicted ones
                if label not in predicted:
                    FP += 1   # False Positive
                    break

    # Similarly for Anomalous files
    with torch.no_grad():
        for line in test_abnormal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1    # True Positive
                    break
    elapsed_time = time.time() - start_time

    # Print the time taken to predict
    print('elapsed_time: {:.3f}s'.format(elapsed_time))

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    f.close()
