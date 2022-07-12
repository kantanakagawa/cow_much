import glob
from numpy import append

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder
df = pd.read_excel("all_data.xlsx")

df = df.loc[df["価格"] < 2000000]
df = df.loc[df["日令"] < 365]


# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(df[["性別"]])  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df[["性別"]])
df["性別"] = t
#############################################################################################################

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(df[["父牛"]])  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
t = label_encoder.transform(df[["父牛"]])
df["父牛"] = t
#############################################################################################################

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

joblib.dump(model, "cow.pkl", compress=True)