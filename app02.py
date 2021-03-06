from flask import Flask, render_template, request
import pandas as pd
import joblib
import warnings

# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


@app.route("/index")
def index():
    return render_template("index.html")


# ユーザー追加のルーティング(POSTでアクセス限定)
@app.route("/cow_much", methods=["POST"])
def cow_much():
    sex = request.form["sex"]
    father = request.form["father"]
    gland = request.form["gland"]
    gege = request.form["gege"]
    got = request.form["got"]
    age = request.form["age"]
    wight = request.form["wight"]

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

    # ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
    enc = LabelEncoder()  # encはencoderの省略名称である変数です。
    # fit()により性別カラムに対してラベルエンコーディングを行います。
    label_encoder = enc.fit(
        df1[["母の祖祖父"]]
    )  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
    # カラムの要素を取得
    # transform()で数値へ変換
    # integer_classes = label_encoder.transform(label_encoder.classes_)
    t = label_encoder.transform(df1[["母の祖祖父"]])
    df1["母の祖祖父"] = t
    #############################################################################################################
    # ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
    enc = LabelEncoder()  # encはencoderの省略名称である変数です。
    # fit()により性別カラムに対してラベルエンコーディングを行います。
    label_encoder = enc.fit(
        df1[["母の祖父"]]
    )  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
    # カラムの要素を取得
    # transform()で数値へ変換
    # integer_classes = label_encoder.transform(label_encoder.classes_)
    t = label_encoder.transform(df1[["母の祖父"]])
    df1["母の祖父"] = t
    #############################################################################################################
    # ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
    enc = LabelEncoder()  # encはencoderの省略名称である変数です。
    # fit()により性別カラムに対してラベルエンコーディングを行います。
    label_encoder = enc.fit(
        df1[["母の父"]]
    )  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
    # カラムの要素を取得
    # transform()で数値へ変換
    # integer_classes = label_encoder.transform(label_encoder.classes_)
    t = label_encoder.transform(df1[["母の父"]])
    df1["母の父"] = t
    #############################################################################################################
    # ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
    enc = LabelEncoder()  # encはencoderの省略名称である変数です。
    # fit()により性別カラムに対してラベルエンコーディングを行います。
    label_encoder = enc.fit(
        df1[["父牛"]]
    )  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
    # カラムの要素を取得
    # transform()で数値へ変換
    # integer_classes = label_encoder.transform(label_encoder.classes_)
    t = label_encoder.transform(df1[["父牛"]])
    df1["父牛"] = t
    #############################################################################################################
    # ラベルエンコーディングを行うために、LabelEncoderクラスをインスタンス化（利用するためのおまじないだとお考えください）
    enc = LabelEncoder()  # encはencoderの省略名称である変数です。
    # fit()により性別カラムに対してラベルエンコーディングを行います。
    label_encoder = enc.fit(
        df1[["性別"]]
    )  # label_encoder = enc.fit(titanic_x["Sex"])  でも可能です
    # transform()で数値へ変換
    # integer_classes = label_encoder.transform(label_encoder.classes_)
    t = label_encoder.transform(df1[["性別"]])
    df1["性別"] = t
    #############################################################################################################

    # 学習済みモデルを読み込み利用します

    # ランダムフォレスト回帰モデルを読み込み
    model = joblib.load("./cow.pkl")
    result = model.predict(df1)
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
