import sys
sys.path.append('../../')
from Log_Parsing import drain_demo
import os
#from logging_support.logger import create_logger

# os.environ['LOG_FILEPATH'] = os.path.join(os.getcwd(), '../logging_support/prod_logs.log')
# os.environ['LOG_LEVEL'] = 'DEBUG'
# logger = create_logger(__name__)

main_dir = 'data/'

raw_logs = main_dir+"All.csv"
#raw_metrics = main_dir+"raw_metrics.csv"

para = {
#***********************************HARD CODED*******************************************************************************************************
    'log_format': '<Content>',  # log format
    'raw_logs': raw_logs,  # path to raw logs
    #'raw_metrics': raw_metrics,  # path to raw metrics
    'input_dir': main_dir,  # The input directory of log file
    'output_dir': main_dir,  # The output directory of parsing results
    'log_file': 'log_data.log',  # The input log file name
    'date_time_in_raw_logs_header': '_source.time',  # column header for date-time stamp in raw logs
    'raw_logs_data_header': '_source.log',  # column header for raw logs data in raw logs file
    'raw_logs_error_header': 'label',  # column header for error in the raw logs file
    'raw_logs_pid_header': '_source.service_id',  # column header for pid in the raw logs file
    'raw_logs_source_logs': main_dir + 'log_data.log',  # path of raw logs souce logs
    'parsed_logs': main_dir + "log_data.log_structured.csv",  # path to parsed logs
    #'parsed_metrics': main_dir + "log_data.kpi.csv",  # path to parsed metrics
    'rep_path':	main_dir,  # path used for savinng all representatives (patterns).
    'date': 'Date',  # date column header
    'time': 'Time',  # time column header
    'lineid': 'LineId',  # lineid/log serial number column header
    'eventid': 'EventId',  # eventID column header
    'pid': 'PID',  # processID column header
    'error_header': 'error',  # error column header
#***********************************HARD CODED*******************************************************************************************************
}

#try:
drain_demo.data_preprocessing(para)
#     logger.info("log parsing successful")
# except Exception as e:Pyt
#     logger.error(e)
#     logger.critical("log parsing failed")