import pandas as pd
import logging


def pre(log_file, txt_file):
    """
                   Parameters
                   ----------
                   log_file : csv file
                       The csv file which is used to produce key sequences for test
                   txt_file : text file
                       The text file that contains the saved eventids for training

                   Description
                   ----------
                   This function reads the Event ids from test dataset and matches them
                   with the existing eventids, if matched, it is kept as the same eventid
                   otherwise, a new eventid is assigned to it.

                   """
    nem = pd.read_csv(log_file)
    ne = nem["EventId"]
    txt_file = open(txt_file, "r")
    a = txt_file.readlines()
    a = str(a)
    a = a.replace("{", "").replace("}", "")
    a = a.split(",")
    l = len(a)
    list1 = []
    for elem in ne:
        for i in range(l):
            ele = a[i]
            ele = ele.replace("'", "").split(":")
            event = ele[0]
            if event == elem:
                pass
            else:
                list1.append(event)
    list1 = set(list1)
    dicti = {}
    k = l
    for i in list1:
        a = i
        key = k
        dicti[a] = k
        k = k + 1
        num_classes = len(dicti)
    nem["EventIDs"] = ""
    nem["EventIDs"] = nem.EventId.map(dicti)
    nem.to_csv(r"data\Test_structured.csv", index=None)
    txt_file.close()

pre(r"data\log_data.log_structured.csv","data\eventids.txt")


