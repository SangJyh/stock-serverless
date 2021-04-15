#!/usr/bin/env python
import pandas as pd
import boto3
import requests
import io
import datetime
import pandas_datareader.data as pdr
import sys
import numpy as np
#from sklearn.preprocessing import RandomForestRegressor
from io import StringIO
# SETUP LOGGING
import logging
from pythonjsonlogger import jsonlogger

#setup
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
LOG.addHandler(logHandler)


### S3 ###
def write_s3(df, bucket):
    """Write S3 Bucket"""

    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource("s3")
    s3_resource.Object(bucket, "stock.csv").put(
        Body=csv_buffer.getvalue()
    )
#    LOG.info(f"result of write to bucket: {bucket} with:\n {res}")
    
#bucket = 'info'  # already created on S3
#csv_buffer = StringIO()
#df.to_csv(csv_buffer)

#s3_resource = boto3.resource('s3')
#s3_resource.Object(bucket, 'df.csv').put(Body=csv_buffer.getvalue())


#ml function
def lambda_handler(event, context):
    """Return a table of 'value' stock price features"""
    #set up the start and end time point I want
    #data preprocessing
        #round numbers
    end = datetime.date.today()
    start = end + datetime.timedelta(days=-365)
    data = pdr.DataReader(event["name"], 'yahoo', start, end)
    data = data.round(3)
    #make model based on the original data
    ##################    
        #add lags information to each observation
    
        
    #################   
    #print(data.head(2))
    #write dataframe into s3 bucket
    print(data.head())
    write_s3(data.head() , "stock.sl605")
    return 

#def modeling(event, data):
#    
#    end = datetime.date.today()
#    start = end + datetime.timedelta(days=-365)
#    
#    n, m = data.shape
#    train = data.iloc[0:(n*3)//4].to_numpy()
#    val = data.iloc[(n*3)//4:n].to_numpy()#[["High", "Low", "Open", "Close", "Volume", "Adj Close"]]
#    features_set = np.delete(train, 3, axis=1).copy() #.append(apple_training_scaled[i, 0:apple_train.shape[1]-1])
#    labels = train[:, 3].copy()
#    features_set_val =  np.delete(val, 3, axis=1).copy() #get the features for training
#    labels_val =  val[:, 3].copy()       #get the prediction result for training
#
##    #data preprocessing
##    data = data.round(3)
##    #make model based on the original data
#    rf = RandomForestRegressor(max_depth=2, random_state=0)
#    rf.fit(features_set, labels)
#    tomorrow = rf.predict(features_set_val)[-1]
#    print("Tomorrow will be:",tomorrow.round(2))
##    tomorrow = lstm(train, val)
#    #change data type and index of the datafram for better visualization
#    data["Volume"] = data.apply(lambda x: "{:,.0f}".format(x["Volume"]), axis=1)
#    data = data.reset_index()
#    
#    #return the table into html format and modify the look
#    #return_table = data.to_html(table_id=stock, justify="center")
#    #return_table = return_table[:6] + " align = 'center'" + return_table[6:]
#    if tomorrow > data["Close"].iloc[-1]:
#        future = 'Next Business Day: Bull'
#    elif tomorrow < data["Close"].iloc[-1]:
#        future = 'Next Business Day: Bear'
#    else: future = 'Next Business Day: Same'
    
#    # add header
#    title = "Based on {0}'s historical stock price (from {1} to {2})".format(event["name"], start, end)
#    result = title + future
#    return result


