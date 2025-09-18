import src.main as main
import pytest

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@pytest.fixture
def data():
    data = main.read_data()
    return data

@pytest.fixture
def df_onehot(data):
    df_onehot =  main.one_hot(data)
    return df_onehot


@pytest.fixture
def target():
    target = "HeartDisease"
    return target

@pytest.fixture
def X(df_onehot, target):
    features = df_onehot.columns[df_onehot.columns != target]
    X = df_onehot[features]
    return X

@pytest.fixture
def y(df_onehot, target):
    y = df_onehot[target]
    return y

@pytest.fixture
def X_train_test(X, y):
    X_train, X_test, y_train, y_test = main.split(X, y)
    return X_train, X_test, y_train, y_test

@pytest.fixture
def X_train(X_train_test):
    return X_train_test[0]

@pytest.fixture
def X_test(X_train_test):
    return X_train_test[1]

@pytest.fixture
def y_train(X_train_test):
    return X_train_test[2]

@pytest.fixture
def y_test(X_train_test):
    return X_train_test[3]

@pytest.fixture
def y_pred(X_train, y_train, X_test):
    return main.model_fit_pred(X_train, y_train, X_test)




# データの読み取り
def test_read_data():
    df = main.read_data()
    assert df.shape == (642,12) ,"正しいデータを読み込めていないです"
# 欠損値を持つ行の削除
def test_nan_drop_record(data:pd.DataFrame):
    df = main.nan_drop_record(data)
    assert df.isna().sum().sum() == 0, "欠損値がまだ存在します" 

# 欠損値を持つ列の削除
def  test_nan_drop_column(data:pd.DataFrame):
    df = main.nan_drop_column(data)
    assert df.isna().sum().sum() == 0, "欠損値がまだ存在します"

# データのエンコード
def test_one_hot(data:pd.DataFrame):
    df = main.one_hot(data)
    assert (df.dtypes == "O").sum() == 0, "オブジェクト属性のカラムがまだあります"

# 目的変数と説明変数の分割
def test_Xy_sepalate(df_onehot:pd.DataFrame, target) -> pd.DataFrame:
    X, y = main.Xy_sepalate(df_onehot, target)
    assert len(X.columns) == len(df_onehot.columns) - 1, "説明変数のカラム数が不正です"
    assert y.equals(df_onehot[target]), "目的変数が不正です"


# 訓練データと検証データに分割
def test_split(X, y):
    X_train, X_test, y_train, y_test = main.split(X, y)
    assert 0.15 <= len(X_test)/len(X) <= 0.25, "分割の比率が0.2通りに分割されませんでした"

# データの標準化
def test_scale(X_train, X_test):
    X_scaled = main.scale(X_train, X_test)
    assert np.abs(np.mean(X_scaled)) <= 0.1, "標準化後の平均が0付近でなく標準化がうまくいっていない可能性があります。"

# モデルの学習から予測値の出力
def test_model_fit_pred(X_train, y_train, X_test):
    y_pred = main.model_fit_pred(X_train, y_train, X_test)
    assert len(y_pred) == len(X_test), "テストデータとレコード数が異なる結果が出力されています"

# 予測値の正解率の出力
def test_accuracy(y_test, y_pred):
    assert 0 <= main.accuracy(y_test, y_pred) <= 1, "正解率が0~1の範囲に存在しません"


