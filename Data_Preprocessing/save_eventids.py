import pandas as pd
import logging

def save(log_file):
    """
                   Parameters
                   ----------
                   log_file : csv file
                       The csv file which is used to produce key sequences


                   Description
                   ----------
                   This function is used to save the eventids in a dictionary
                   which can be used to generate sequences for test data.

                   """
    ne = pd.read_csv(log_file)
    list1 = ne["EventId"]
    list1 = set(list1)
    len(list1)
    dicti = {}
    k = 1
    for i in list1:
        a = i
        key = k
        dicti[a] = k
        k = k + 1
    num_classes = len(dicti)

    ne["EventIDs"] = ""
    ne["EventIDs"] = ne.EventId.map(dicti)
    ne.to_csv(r"data\Logs_structured.csv", index=None)
    try:
        eid = open('data\eventids.txt', 'wt')
        eid.write(str(dicti))
        eid.close()

    except:
        logging.warning("Unable to save the eventids")


save(r"data\log_data.log_structured.csv")