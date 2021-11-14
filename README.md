# DeepLog
Anomaly detection is a critical step towards building a secure and trustworthy system. The primary purpose of a system log is to record system states and significant events at various critical points to help debug system failures and perform root cause analysis. Such log data is universally available in nearly all computer systems. Log data is an important and valuable resource for understanding system status and performance issues; therefore, the various system logs are naturally excellent source of information for online monitoring and anomaly detection. We propose DeepLog, a deep neural network model utilizing Long Short-Term Memory (LSTM), to model a system log as a natural language sequence. This allows DeepLog to automatically learn log patterns from normal execution, and detect anomalies when log patterns deviate from the model trained from log data under normal execution.

## LOG KEY EXECUTION PATH ANOMALY DETECTION

This is an attempt to create the first part of the paper **DeepLog: Anomaly Detection and Diagnosis from System Logs
through Deep Learning**.

**INSTALLING**

Download the project with:<br>
git clone https://github.com/PragyaShukla2580/DeepLog.git <br>
For training the model:<br>
python LogKeyModel_train.py
<br> For getting the predictions<br>
python LogKeyModel_predict.py</li></ol>

**STRUCTURE**

This project involves three steps:
<ol><li><b>Data Preprocessing</b>:<br>
For Data Preprocessing, we will make use of Log Parser.<br>
For that, we will collect the entire data in the form of csv file, and 
and run logparser.py from Data_Preprocessing. This will
convert the unstructured logs to structured data with the help of Drain parser.
</li>
<li><b>Feature Engineering</b>:<br>
On the structured file obtained, we will do Feature Engineering.<br>
For that, we will run feature_engg.py and will get a file called Train.csv which contains all the normal logs to train the model in 
the required format. We will call the same function with test file, and will get the test file also in the required format.
</li>
<li><b>Model Training and Prediction</b>:<br>
For model training, we will use Train.csv file to train and for prediction, we will use Test_Out.csv file, which was created by calling Feature Engineering on the File that had some anomalous values.
</li></ol>

# Achievements

Successfully Completed this project.

![img.png](img.png)
