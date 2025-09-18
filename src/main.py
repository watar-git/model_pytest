import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# データの読み取り
def read_data():
    df = pd.read_csv("data/train.csv")
    return df
# 欠損値を持つ行の削除
def nan_drop_record(data:pd.DataFrame):
    data_dropnah = data.dropna(axis=0)
    return data_dropnah

# 欠損値を持つ列の削除
def  nan_drop_column(data:pd.DataFrame):
    data_dropv = data.dropna(axis=1)
    return data_dropv

# データのエンコード
def one_hot(data:pd.DataFrame):
    object_columns = data.columns[data.dtypes == "O"]
    df_hot = pd.get_dummies(data,columns=object_columns)
    return df_hot

# 目的変数と説明変数の分割
def Xy_sepalate(data:pd.DataFrame, target):
    features = data.columns[data.columns != target]

    X = data[features]
    y = data[target]
    return X, y

# 訓練データと検証データに分割
def split(X,y):
    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test

# データの標準化
def scale(fit_X, transform_X):
    scaler = StandardScaler()
    scaler.fit(fit_X)
    X_scaled = scaler.transform(transform_X)
    return X_scaled

# モデルの学習から予測値の出力
def model_fit_pred(X_train, y_train, X_test):
    lr = LogisticRegression(penalty="l2", C=0.43)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    return y_pred

# 予測値の正解率の出力
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)