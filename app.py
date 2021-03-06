from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators
import joblib
import pandas as pd
import warnings
# 警告文を非表示
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder


# 学習済みモデルを読み込み利用します
def praice(df1):
    # ランダムフォレスト回帰モデルを読み込み
    model = joblib.load("./cow.pkl")
    pred = model.predict(df1)
    return pred


app = Flask(__name__)
# app.config.from_object(__name__)
# app.config["SECRET_KEY"] = "zJe09C5c3tMf5FnNL09C5d6SAzZoY"


class cowform(Form):
    sex = (
        "性別",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    father = (
        "1代祖",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    gland = (
        "2代祖",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    gege = (
        "3代祖",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    got = (
        "4代祖",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    age = (
        "日齢",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )

    wight = (
        "体重",
        [
            validators.InputRequired("この項目は入力必須です"),
        ],
    )
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
    integer_classes = label_encoder.transform(label_encoder.classes_)
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
    integer_classes = label_encoder.transform(label_encoder.classes_)
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
    integer_classes = label_encoder.transform(label_encoder.classes_)
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
    integer_classes = label_encoder.transform(label_encoder.classes_)
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
    pred = praice(x)
    praice(pred)
    return render_template("result.html")
    # elif request.method == 'GET':

    #     return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)
