import numpy as np
import pandas as pd
import torch
import transformers

from flask import Flask, request, render_template, redirect, url_for
import logging
from datetime import datetime
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

from transformers import AlbertTokenizer, AlbertForSequenceClassification




import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(text):

    # 保存したパスからモデルとトークナイザーをロード
    model_path = './saved_model'
    loaded_tokenizer = AlbertTokenizer.from_pretrained(model_path)
    loaded_model = AlbertForSequenceClassification.from_pretrained(model_path)

    # モデルがGPUにある場合、CPUに移動
    loaded_model.cpu()

    # 推論の実行
    inputs = loaded_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        app.logger.debug("現在の日時11: %s", datetime.now())
        outputs = loaded_model(**inputs)
        app.logger.debug("現在の日時12: %s", datetime.now())
        logits = outputs.logits
        app.logger.debug("現在の日時13: %s", datetime.now())
        predicted_label = torch.argmax(logits, dim=1)  # 最も確率が高いラベルを取得
        app.logger.debug("現在の日時14: %s", datetime.now())
        print(predicted_label)

    return predicted_label


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            #wisper AIに食べさせられるように音声ファイルをrb形でopenしておく
            with open(file_path, "rb") as f:
                #wisper AIに音源データを文字起こしさせる
                #transcription = openai.Audio.transcribe("whisper-1", f)
                #txt = transcription['text']
                txt = "This is a pen"
            #ここで取得した音声データの文字データtxtをベースに推論を実施

            from datetime import datetime
            app.logger.debug("現在の日時1s: %s", datetime.now())
            #pred = 0
            pred = predict(txt)
            app.logger.debug("現在の日時1e: %s", datetime.now())

        return render_template('index.html')

    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
