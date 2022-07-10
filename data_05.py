import glob
from numpy import append

import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

df = pd.read_excel("all_data.xlsx")


sex = input("性別を入力してください:")
father = input("1代祖を入力してください:")
gland = input("2代祖を入力してください:")
gege = input("3代祖を入力してください:")
got = input("4代祖を入力してください:")
age = input("日齢を入力してください:")
wight = input("体重を入力してください:")

df1 = pd.DataFrame(
    data={
        "性別": [sex],
        "父牛": [father],
        "母の父": [gland],
        "母の祖父": [gege],
        "母の祖祖父": [got],
        "日令": [age],
        "体重": [wight],
    }
)

df = df.loc[df["価格"] < 2000000]
df = df.loc[df["日令"] < 365]

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(df[["性別"]])  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df[["性別"]])
df["性別"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(df[["父牛"]])  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df[["父牛"]])
df["父牛"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(
    df[["母の父"]]
)  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# カラムの要素を取得
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df[["母の父"]])
df["母の父"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(
    df[["母の祖父"]]
)  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df[["母の祖父"]])
df["母の祖父"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(
    df[["母の祖祖父"]]
)  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# カラムの要素を取得
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df[["母の祖祖父"]])
df["母の祖祖父"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(
    df1[["母の祖祖父"]]
)  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# カラムの要素を取得
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df1[["母の祖祖父"]])
df1["母の祖祖父"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(
    df1[["母の祖父"]]
)  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# カラムの要素を取得
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df1[["母の祖父"]])
df1["母の祖父"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(
    df1[["母の父"]]
)  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# カラムの要素を取得
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df1[["母の父"]])
df1["母の父"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(
    df1[["父牛"]]
)  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# カラムの要素を取得
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df1[["父牛"]])
df1["父牛"] = t
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(
    df1[["性別"]]
)  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df1[["性別"]])
df1["性別"] = t
#############################################################################################################


X = df[["性別", "父牛", "母の父", "母の祖父", "母の祖祖父", "日令", "体重"]].values
y = df["価格"].values

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=42
)
best_model = ""
pred_model = []
#  RandomForest回帰
model = RandomForestRegressor(100)  # ランダムフォレスト回帰モデル
model.fit(train_X, train_y)  # 学習
pred_y = model.predict(test_X)  # 予測
mse = mean_squared_error(test_y, pred_y)  # 評価
print(model.predict(df1))

# 去勢	愛之国	安糸福	福谷桜	金幸	287	345	603000
# 去勢	諒太郎	第２平茂勝	照美	福茂	282	354	717200