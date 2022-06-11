import pandas as pd
import pickle

from prefect import flow, task, get_run_logger

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

import os
import re

def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def get_paths(data:None):
    train_path = ''
    test_path = ''
    form = '%Y-%m'
    if data==None:
        data = datetime.now().strftime(form)
        train_time = data - relativedelta(months=2)
        val_time = data - relativedelta(months=1)
        matched_date = {}
        
        for k in os.listdir('./data'):
            match = re.search(r"[0-9]{4}\-[0-9]{2}", k)
            if match.group(0) != None:
                matched_date[match.group(0)] = k
        for key,value in matched_date.items():
            if datetime.strptime(key,form).strftime(form) == train_time.strftime(form):
                train_path = f"./data/{value}"
            if datetime.strptime(key,form).strftime(form) == val_time.strftime(form):
                test_path = f"./data/{value}"
            else:
                continue
    else:
        data = datetime.strptime(data,"%Y-%m-%d")
        train_time = data - relativedelta(months=2)
        val_time = data - relativedelta(months=1)
        matched_date = {}
        
        for k in os.listdir('./data'):
            match = re.search(r"[0-9]{4}\-[0-9]{2}", k)
            if match.group(0) != None:
                matched_date[match.group(0)] = k
        for key,value in matched_date.items():
            if datetime.strptime(key,form).strftime(form) == train_time.strftime(form):
                train_path = f"./data/{value}"
            if datetime.strptime(key,form).strftime(form) == val_time.strftime(form):
                test_path = f"./data/{value}"
            else:
                continue
        
        return train_path, test_path
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        get_run_logger(f"The mean duration of training is {mean_duration}")
    else:
        get_run_logger(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def train_model(df, categorical,date:str):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    get_run_logger(f"The shape of X_train is {X_train.shape}")
    get_run_logger(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    get_run_logger(f"The MSE of training is: {mse}")

    with open(f'model-{date}.bin','wb') as model_out:
        pickle.dump(lr, model_out)
    with open(f'dv-{date}.b','wb') as dv_out:
        pickle.dump(dv, dv_out)

    return lr, dv

def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    get_run_logger(f"The MSE of validation is: {mse}")
    return
@flow
def main(date="2021-03-15"):
    # train_path: str = './data/fhv_tripdata_2021-01.parquet', 
    # val_path: str = './data/fhv_tripdata_2021-02.parquet
    categorical = ['PUlocationID', 'DOlocationID']
    train_path, val_path = get_paths(date).result()
    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical,date)
    run_model(df_val_processed, categorical, dv, lr)

# main(date="2021-03-15")


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow = main,
    name = "model training",
    schedule = CronSchedule(cron = "0 9 15 * *"),
    flow_runner = SubprocessFlowRunner(),
    tags = ['assignment'],
)

