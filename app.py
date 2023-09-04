import numpy as np
import pandas as pd
import torch
import transformers

from flask import Flask, request, render_template, redirect, url_for
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
import logging
from datetime import datetime

from transformers import AlbertTokenizer, AlbertForSequenceClassification

# 保存したパスからモデルとトークナイザーをロード
model_path = './saved_model'
loaded_tokenizer = AlbertTokenizer.from_pretrained(model_path)
loaded_model = AlbertForSequenceClassification.from_pretrained(model_path)

# モデルがGPUにある場合、CPUに移動
loaded_model.cpu()

# 推論の実行
inputs = loaded_tokenizer("サンプルテキスト", return_tensors="pt")
with torch.no_grad():
    app.logger.debug("現在の日時11: %s", datetime.now())
    outputs = loaded_model(**inputs)
    app.logger.debug("現在の日時12: %s", datetime.now())
    logits = outputs.logits
    app.logger.debug("現在の日時13: %s", datetime.now())
    predicted_label = torch.argmax(logits, dim=1)  # 最も確率が高いラベルを取得
    app.logger.debug("現在の日時14: %s", datetime.now())
    print(predicted_label)







@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
