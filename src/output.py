import pandas as pd
import json
import logging
from collections import OrderedDict


def func(log_file, p_file, window='session'):
    """
                Parameters
                ----------
                log_file : csv file
                p_file: csv file


                Description
                ----------
                This function reads the two files and return the timestamp
                of the detected anomalies in json file.

                """
    re = pd.read_csv(p_file)
    l = len(re)
    p = {}
    j = 0
    for i in range(l):
        window_size = 9
        j = j + 1
        eventList = re["Pred_Sequence"][i].replace("[", "").replace("]", "").split(", ")
        l1 = len(eventList)
        p[j] = []
        for k in range(l1):
            window_size += 1
            if eventList[k] != "" and eventList[k] != "0":
                p[j].append(window_size)

    EIDnormal = []
    EIDabnormal = []
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    log_file = pd.read_csv(log_file)
    data_dict1 = OrderedDict()
    data_dict2 = OrderedDict()
    k = 0
    for ids, row in log_file.iterrows():
        EIDnormal.append(row["PID"])
        EIdnormal = set(EIDnormal)
    temp = log_file.groupby("PID")
    for x, df in temp:
        k = k + 1
        data_dict1[k] = []
    k = 0
    for x, df in temp:
        k = k + 1
        data_dict2[k] = []
    k = 0
    for x, df in temp:
        k = k + 1

        c = list(x for x in df["_source.time"])
        data_dict1[k].append(c)

    k = 0
    for x, df in temp:
        k = k + 1

        d = list(x for x in df["_id"])
        data_dict2[k].append(d)

    timestamp = []
    line = 0
    for i in p.keys():
        list1 = p[i]
        if len(list1) != 0:
            for ele in list1:
                if ele != list1[-1]:
                    timestamp.append(data_dict1[i][0][ele - 1])

    pids = []
    line = 0
    for i in p.keys():
        list1 = p[i]
        if len(list1) != 0:
            for ele in list1:
                if ele != list1[-1]:
                    pids.append(data_dict2[i][0][ele - 1])

    dictionary = {
        "Application Details": {
            "Name of the application": [_id for _id in
                                        list(set(list(log_file['_source.kubernetes.namespace_name'])[:5])) if
                                        str(_id) != 'nan'],
            "Name of the microservice(s)": [_id for _id in
                                            list(set(list(log_file['_source.kubernetes.container_name'])[:5]))
                                            if str(_id) != 'nan'],
            "Timestamp": [t for t in list(set(list(timestamp)[:])) if str(t) != 'nan'],
            "Log IDs": [l for l in list(set(list(pids)[:])) if str(l) != 'nan']
        }
    }
    jsonString = json.dumps(dictionary)
    data = pd.read_json(jsonString)
    data.to_json(r"data\output.json")


func(r"data\Test.csv", r"data\output.csv")
