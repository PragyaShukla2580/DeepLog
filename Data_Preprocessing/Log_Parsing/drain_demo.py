#!/usr/bin/env python
#import sys

#sys.path.append('../../')
from . import drain


def data_preprocessing(para):
    # Regular expression list for optional preprocessing (default: [])
    regex = [
        r'blk_(|-)[0-9]+',  # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
    ]
    st = 0.5  # Similarity threshold
    depth = 4  # Depth of all leaf nodes

    preprocessing = drain.Processing(para['raw_logs'], para['date_time_in_raw_logs_header'],
                                     para['date'], para['time'], para['raw_logs_data_header'],
                                     para['raw_logs_error_header'], para['raw_logs_pid_header'],
                                     para['raw_logs_source_logs'], para['parsed_logs'],
                                     para['error_header'], para['pid'])
    preprocessing.preprocessing()
    parser = drain.LogParser(para['log_format'], indir=para['input_dir'], outdir=para['output_dir'], depth=depth, st=st,
                             rex=regex)
    parser.parse(para['log_file'])
    preprocessing.postprocessing()
   # preprocessing.cleanup_metrics()

    return (print("Data Preprocessing complete"))