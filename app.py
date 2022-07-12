from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators
import joblib
import pandas as pd


# 学習済みモデルを読み込み利用します
def praice(df1):
    # ランダムフォレスト回帰モデルを読み込み
    model = joblib.load("./cow.pkl")
    pred = model.predict(df1)
    return pred


app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_object(__name__)
app.config["SECRET_KEY"] = "zJe09C5c3tMf5FnNL09C5d6SAzZoY"


class cowform(Form):
    sex = FloatField(
        "性別",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    father = FloatField(
        "1代祖",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    gland = FloatField(
        "2代祖",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    gege = FloatField(
        "3代祖",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    god = FloatField(
        "4代祖",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    age = FloatField(
        "日齢",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    wight = FloatField(
        "体重",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )


# 公式サイト
# http://wtforms.simplecodes.com/docs/0.6/fields.html
# Flaskとwtformsを使い、index.html側で表示させるフォームを構築します。


# html側で表示するsubmitボタンの表示
submit = SubmitField("判定")

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

@app.route("/cow_much", methods=["GET", "POST"])
def predicts():
    form = cowform(request.form)
    if request.method == "POST":
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template("index.html", form=form)
        return render_template("index.html", form=form)
    else:
        sex = int(request.form["sex"])
        father = int(request.form["father"])
        gland = int(request.form["gland"])
        gege = int(request.form["gege"])
        got = int(request.form["got"])
        age = int(request.form["age"])
        wight = int(request.form["wight"])
        x = df1([sex, father, gland, gege, got, age, wight])
    pred = praice(x)
    praice(pred)
    return render_template("result.html")
    # elif request.method == 'GET':

    #     return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template, request, flash
# from wtforms import Form, FloatField, SubmitField, validators, ValidationError
# import numpy as np
# from sklearn.externals import joblib

# # 学習済みモデルを読み込み利用します
# def predict(parameters):
#     # ニューラルネットワークのモデルを読み込み
#     model = joblib.load("./nn.pkl")
#     params = parameters.reshape(1, -1)
#     pred = model.predict(params)
#     return pred


# # ラベルからIrisの名前を取得します
# def getName(label):
#     print(label)
#     if label == 0:
#         return "Iris Setosa"
#     elif label == 1:
#         return "Iris Versicolor"
#     elif label == 2:
#         return "Iris Virginica"
#     else:
#         return "Error"


# app = Flask(__name__)
# app.config.from_object(__name__)
# app.config["SECRET_KEY"] = "zJe09C5c3tMf5FnNL09C5d6SAzZoY"

# # 公式サイト
# # http://wtforms.simplecodes.com/docs/0.6/fields.html
# # Flaskとwtformsを使い、index.html側で表示させるフォームを構築します。
# class IrisForm(Form):
#     SepalLength = FloatField(
#         "Sepal Length(cm)（蕚の長さ）",
#         [
#             validators.InputRequired("この項目は入力必須です"),
#             validators.NumberRange(min=0, max=10),
#         ],
#     )

#     SepalWidth = FloatField(
#         "Sepal Width(cm)（蕚の幅）",
#         [
#             validators.InputRequired("この項目は入力必須です"),
#             validators.NumberRange(min=0, max=10),
#         ],
#     )

#     PetalLength = FloatField(
#         "Petal length(cm)（花弁の長さ）",
#         [
#             validators.InputRequired("この項目は入力必須です"),
#             validators.NumberRange(min=0, max=10),
#         ],
#     )

#     PetalWidth = FloatField(
#         "petal Width(cm)（花弁の幅）",
#         [
#             validators.InputRequired("この項目は入力必須です"),
#             validators.NumberRange(min=0, max=10),
#         ],
#     )

#     # html側で表示するsubmitボタンの表示
#     submit = SubmitField("判定")


# @app.route("/", methods=["GET", "POST"])
# def predicts():
#     form = IrisForm(request.form)
#     if request.method == "POST":
#         if form.validate() == False:
#             flash("全て入力する必要があります。")
#             return render_template("index.html", form=form)
#         else:
#             SepalLength = float(request.form["SepalLength"])
#             SepalWidth = float(request.form["SepalWidth"])
#             PetalLength = float(request.form["PetalLength"])
#             PetalWidth = float(request.form["PetalWidth"])

#             x = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth])
#             pred = predict(x)
#             irisName = getName(pred)

#             return render_template("result.html", irisName=irisName)
#     elif request.method == "GET":

#         return render_template("index.html", form=form)


# if __name__ == "__main__":
#     app.run()
