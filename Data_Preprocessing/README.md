# DATA PREPROCESSING
This code is available at CloudAEye utils.

For Log Parsing, we are using 
 
1. https://github.com/CloudAEye/incident-detection/tree/main/demo.
Here, we will use log_parser.py to parse the logs.
   
2. https://github.com/CloudAEye/cloudaeye-utils/blob/master/src/cloudaeye/data_engineering/logs/drain.py and https://github.com/CloudAEye/incident-detection/blob/main/log_parsing/drain_demo.py are internally called by log_parser.py.

3. The only difference between the existing code and requirements for Deeplog is that, in the Log Parsing, we will not take the use of raw_metrics, and wherever raw_metrics are used, we will comment that out.