import glob

# from statistics import LinearRegression
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


df = pd.read_excel("all_data.xlsx")

# sns.distplot(df["価格"])
# plt.show()


import warnings

# 警告文を非表示
warnings.simplefilter("ignore")

from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(df[["性別"]])  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# カラムの要素を取得
print("Categorical classes:", label_encoder.classes_)
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
print("Integer classes:", integer_classes)

t = label_encoder.transform(df[["性別"]])
df["性別"] = t

# ラベルエンコーディングされたことを確認
print(df[["性別"]])
#############################################################################################################

import warnings

# 警告文を非表示
warnings.simplefilter("ignore")

from sklearn.preprocessing import LabelEncoder

# ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
enc = LabelEncoder()  # encはencoderの省略名称である変数です。
# fit()により性別カラムに対してラベルエンコーディングを行います。
label_encoder = enc.fit(df[["父牛"]])  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
# カラムの要素を取得
print("Categorical classes:", label_encoder.classes_)
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
print("Integer classes:", integer_classes)

t = label_encoder.transform(df[["父牛"]])
df["父牛"] = t

# ラベルエンコーディングされたことを確認
print(df[["父牛"]])
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
print("Categorical classes:", label_encoder.classes_)
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
print("Integer classes:", integer_classes)

t = label_encoder.transform(df[["母の父"]])
df["母の父"] = t

# ラベルエンコーディングされたことを確認
print(df[["母の父"]])
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
# カラムの要素を取得
print("Categorical classes:", label_encoder.classes_)
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
print("Integer classes:", integer_classes)

t = label_encoder.transform(df[["母の祖父"]])
df["母の祖父"] = t

# ラベルエンコーディングされたことを確認
print(df[["母の祖父"]])
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
print("Categorical classes:", label_encoder.classes_)
# transform()で数値へ変換
integer_classes = label_encoder.transform(label_encoder.classes_)
print("Integer classes:", integer_classes)

t = label_encoder.transform(df[["母の祖祖父"]])
df["母の祖祖父"] = t

# ラベルエンコーディングされたことを確認
print(df[["母の祖祖父"]])
#############################################################################################################

# print(df.info())


X = df[["性別", "父牛", "母の父", "母の祖父", "母の祖祖父", "日令", "体重"]].values
y = df["価格"].values

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.5, random_state=42
)

best_model = ""
pred_model = []

# 線形回帰
model = LinearRegression()  # 線形回帰モデル
model.fit(train_X, train_y)  # 学習
pred_y = model.predict(test_X)  # 予測
mse = mean_squared_error(test_y, pred_y)  # 評価
print("線形RMSE : %.2f" % (mse**0.5))
min_mse = mse
best_model = "線形"
pred_model = pred_model.append(model)
#############################################################################################################

# リッジ回帰
model = Ridge()  # リッジ回帰モデル
model.fit(train_X, train_y)  # 学習
pred_y = model.predict(test_X)  # 予測
mse = mean_squared_error(test_y, pred_y)  # 評価
print("リッジRMSE : %.2f" % (mse**0.5))
if min_mse > mse:
    min_mse = mse
    best_model = "リッジ"
    pred_model = model
#############################################################################################################

# ラッソ回帰
model = Lasso()  # ラッソ回帰モデル
model.fit(train_X, train_y)  # 学習
pred_y = model.predict(test_X)  # 予測
mse = mean_squared_error(test_y, pred_y)  # 評価
print("ラッソRMSE : %.2f" % (mse**0.5))

if min_mse > mse:
    min_mse = mse
    best_model = "ラッソ"
    pred_model = model
    #############################################################################################################

#  ElasticNet回帰
model = ElasticNet(l1_ratio=0.5)  # エラスティックネット回帰モデル
model.fit(train_X, train_y)  # 学習
pred_y = model.predict(test_X)  # 予測
mse = mean_squared_error(test_y, pred_y)  # 評価
print("エラスティックネットRMSE : %.2f" % (mse**0.5))

if min_mse > mse:
    min_mse = mse
    best_model = "エラスティックネット"
    pred_model = model
    #############################################################################################################

    #  RandomForest回帰
model = RandomForestRegressor(100)  # ランダムフォレスト回帰モデル
model.fit(train_X, train_y)  # 学習
pred_y = model.predict(test_X)  # 予測
mse = mean_squared_error(test_y, pred_y)  # 評価
print("ランダムフォレストRMSE : %.2f" % (mse**0.5))

if min_mse > mse:
    min_mse = mse
    best_model = "ランダムフォレスト回帰"
    pred_model = model
    #############################################################################################################
    plt.figure()
plt.scatter(train_y, model.predict(train_X), label="Train", c="blue")
plt.scatter(test_y, pred_y, c="lightgreen", label="Test", alpha=0.8)
plt.title("Predictor")
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.show()

# sns.set(font='')
# cols = [
#     "価格",
#     "父牛",
#     "母の父",
#     "母の祖父",
#     "母の祖祖父",
# ]
# a = sns.pairplot(df[cols], height=2.5)
