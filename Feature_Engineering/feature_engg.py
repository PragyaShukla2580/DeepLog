import pandas as pd
import logging
from collections import OrderedDict


def func(log_file, window='session'):
    """
                Parameters
                ----------
                log_file : csv file
                    The csv file which is used to produce key sequences


                Description
                ----------
                This function reads the eventids and convert them into
                event sequences. This function is used to generate both
                train and test files.

                """
    EIDnormal = []
    EIDabnormal = []
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    log_file = pd.read_csv(log_file)
    data_dict1 = OrderedDict()
    data_dict2 = OrderedDict()
    for ids, row in log_file.iterrows():
        EIDnormal.append(row['PID'])
        EIdnormal = set(EIDnormal)
        length = len(EIdnormal)
    temp = log_file.groupby("PID")
    for x,df in temp:
        data_dict1[x] = []
    for x,df in temp:
        b = list(x for x in df["EventIDs"])
        data_dict1[x].append(b)
    data_df = pd.DataFrame(list(data_dict1.items()), columns=['BlockId', 'EventSequence'])
    logging.debug("Event sequences created")
    total = []
    a=data_df["EventSequence"]
    a=pd.DataFrame(data=a)
    for i,row in a.iterrows():
        s = ""
        eventList = row.replace("'", "").replace(",", "")
        eList = ""
        for element in eventList:
            num = element
            eList += " " + str(num)
        eList = eList.replace("[", "").replace("]", "").replace(",", "")
        eList = eList.strip()
        total.append([eList])
    dfnew = pd.DataFrame(total, columns=['Sequence'])
    sequence_normal = dfnew["Sequence"]

    #####  UNIT TEST #######

    #### TEST 1 ####

    try:
        for ele in sequence_normal:
            s = ele.split(" ")
            for i in s:
                if type(int(i)) == int:
                    pass
    except TypeError:
        logging.critical("Sequences should be in integer format")

    #### TEST 2 ####

    try:
        if len(sequence_normal) == length:
            pass
    except Exception:
        logging.critical("Rows in sequences are not equal to number of processes")

    sequence_normal.to_csv(r"data\Train.csv", index=None,header=None)
    logging.info("Event sequences saved")
    # except Exception:
    #     logging.critical("Feature Engineering stopped")


if __name__ == '__main__':
    try:
        func(r"data\Logs_structured.csv")
    except Exception:
        logging.critical("Can't preprocess data")