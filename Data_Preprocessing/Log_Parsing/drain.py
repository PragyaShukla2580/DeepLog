"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import regex as re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
import pytz

class Logcluster:
    def __init__(self, logTemplate='', logIDL=None):
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken

class Processing:
    def __init__(self, raw_logs, date_time_in_raw_logs_header, date_header, time_header,
        raw_logs_data_header, raw_logs_error_header, raw_logs_pid_header, raw_logs_source_logs, parsed_logs, error_header, pid_header):
        self.raw_logs = raw_logs
        #self.raw_metrics = raw_metrics
        self.date_time_in_raw_logs_header = date_time_in_raw_logs_header
        self.date_header = date_header
        self.time_header = time_header
        self.raw_logs_data_header = raw_logs_data_header
        self.raw_logs_error_header = raw_logs_error_header
        self.raw_logs_pid_header = raw_logs_pid_header
        self.raw_logs_source_logs = raw_logs_source_logs
        self.parsed_logs = parsed_logs
  #      self.parsed_metrics = parsed_metrics
        self.error_header = error_header
        self.pid_header = pid_header
        self.date = 0
        self.time = 0
        self.error = 0
        self.pid = 0

    def data_cleanup(self):
        logs_parsed = pd.read_csv(self.raw_logs)
        metrics_parsed = pd.read_csv(self.raw_metrics)
        metrics_parsed['timestamp1'] = metrics_parsed['timestamp'].apply(lambda x: datetime.fromtimestamp(x).astimezone(pytz.utc))
        logs_parsed['_source.time1'] = pd.to_datetime(logs_parsed['_source.time'])
        logs_parsed['_source.time1'] = logs_parsed['_source.time1'].dt.floor('T')
        unique_log_timestamps = logs_parsed['_source.time1'].drop_duplicates()
        metrics_parsed = metrics_parsed[metrics_parsed['timestamp1'].isin(unique_log_timestamps)].loc[:, ~metrics_parsed.columns.str.contains('^Unnamed')]
        logs_parsed = logs_parsed[logs_parsed['_source.time1'].isin(metrics_parsed['timestamp1'])].loc[:, ~logs_parsed.columns.str.contains('^Unnamed')]
        metrics_parsed = metrics_parsed.drop(columns=['timestamp1'], axis=1)
        logs_parsed = logs_parsed.drop(columns=['_source.time1'], axis=1)

        logs_parsed.to_csv(self.parsed_logs, index=False)
        metrics_parsed.to_csv(self.parsed_metrics, index=False)

    def preprocessing(self):
        original_logs = pd.read_csv(self.raw_logs)
        original_logs = original_logs.replace(r'\\n',' ', regex=True)
        original_logs = original_logs.sort_values(by=self.date_time_in_raw_logs_header)
        original_logs = original_logs.reset_index(drop=True)
        original_logs[self.date_header] = [pd.to_datetime(d, format='%Y-%m-%d %H:%M:%S.%f').date() for d in original_logs[self.date_time_in_raw_logs_header]]
        original_logs[self.time_header] = [pd.to_datetime(d, format='%Y-%m-%d %H:%M:%S.%f').time() for d in original_logs[self.date_time_in_raw_logs_header]]
        source_logs = list(original_logs[self.raw_logs_data_header])
        self.date = list(original_logs[self.date_header])
        self.time = list(original_logs[self.time_header])
        try:
            self.error = list(original_logs[self.raw_logs_error_header])
        except:
            pass
        self.pid = list(original_logs[self.raw_logs_pid_header])
        original_logs = original_logs.drop([self.date_header, self.time_header], axis=1)
        #original_metrics = pd.read_csv(self.raw_metrics)
        #original_metrics.to_csv(self.parsed_metrics)
        with open(self.raw_logs_source_logs, 'w') as f:
            for item in source_logs:
                f.write(item.replace('\r', '').replace('\n', ' ') + "\n")

        return(print("\n preprocessing completed \n"))

    def postprocessing(self):
        original_logs = pd.read_csv(self.raw_logs)
        original_logs = original_logs.sort_values(by=self.date_time_in_raw_logs_header)
        original_logs = original_logs.reset_index(drop=True)
        logs_parsed = pd.read_csv(self.parsed_logs)
        logs_parsed[self.date_header] = self.date
        logs_parsed[self.time_header] = self.time
        try:
            logs_parsed[self.error_header] = self.error
        except:
            pass
        logs_parsed[self.pid_header] = self.pid
        logs_parsed = pd.concat([logs_parsed, original_logs], axis=1)
        logs_parsed.to_csv(self.parsed_logs)

        return(print("\n postprocessing completed \n"))

    def cleanup_metrics(self):
        logs_parsed = pd.read_csv(self.parsed_logs)
        metrics_parsed = pd.read_csv(self.parsed_metrics)
        metrics_parsed['timestamp1'] = metrics_parsed['timestamp'].apply(lambda x: datetime.fromtimestamp(x).astimezone(pytz.utc))
        logs_parsed['_source.time1'] = pd.to_datetime(logs_parsed['_source.time'])
        logs_parsed['_source.time1'] = logs_parsed['_source.time1'].dt.floor('T')
        unique_log_timestamps = logs_parsed['_source.time1'].drop_duplicates()
        metrics_parsed = metrics_parsed[metrics_parsed['timestamp1'].isin(unique_log_timestamps)].loc[:, ~metrics_parsed.columns.str.contains('^Unnamed')]
        logs_parsed = logs_parsed[logs_parsed['_source.time1'].isin(metrics_parsed['timestamp1'])].loc[:, ~logs_parsed.columns.str.contains('^Unnamed')]
        metrics_parsed = metrics_parsed.drop(columns=['timestamp1'], axis=1)
        logs_parsed = logs_parsed.drop(columns=['_source.time1'], axis=1)

        logs_parsed.to_csv(self.parsed_logs, index=False)
        metrics_parsed.to_csv(self.parsed_metrics, index=False)

    def cleanup_metrics_old(self):
        logs_parsed = pd.read_csv(self.parsed_logs)
        metrics_parsed = pd.read_csv(self.parsed_metrics)
        metrics_parsed['timestamp'] = metrics_parsed['timestamp'].apply(lambda x: datetime.fromtimestamp(x).astimezone(pytz.utc))
        logs_parsed['_source.time'] = pd.to_datetime(logs_parsed['_source.time'])
        grp_by = logs_parsed.groupby(pd.Grouper(key='_source.time', freq='1min'))
        grped = grp_by.groups.keys()
        grped = list(map(lambda x: str(x), grped))
        metrics_parsed = metrics_parsed[metrics_parsed['timestamp'].isin(grped)].loc[:, ~metrics_parsed.columns.str.contains('^Unnamed')]
        metrics_parsed.to_csv(self.parsed_metrics)

class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4,
                 maxChild=100, rex=[], keep_para=True):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None

        seqLen = len(seq)
        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return retLogClust
            currentDepth += 1

        logClustL = parentn.childD

        retLogClust = self.fastMatch(logClustL, seq)

        return retLogClust

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:

            #Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            #If token not matched in this layer of existing tree.
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD)+1 < self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD)+1 == self.maxChild:
                            newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']

                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth+1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']

            #If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    #seq1 is template
    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar


    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim>maxSim or (curSim==maxSim and curNumOfPara>maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            template_str = ' '.join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)


        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False, columns=["EventId", "EventTemplate", "Occurrences"])


    def printTree(self, node, dep):
        pStr = ''
        for i in range(dep):
            pStr += '\t'

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep+1)


    def parse(self, logName):
        print(('Parsing file: ' + os.path.join(self.path, logName)))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []

        self.load_data()

        count = 0
        for idx, line in self.df_log.iterrows():
            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            # logmessageL = filter(lambda x: x != '', re.split('[\s=:,]', self.preprocess(line['Content'])))
            matchCluster = self.treeSearch(rootNode, logmessageL)

            #Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            #Add the new log message to the existing cluster
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                    matchCluster.logTemplate = newTemplate

            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print(('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log))))


        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.outputResult(logCluL)

        print(('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time)))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\s+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list