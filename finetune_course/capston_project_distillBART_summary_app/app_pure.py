# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:35:44 2025

@author: 10543
"""

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("NuppuCat/distillBart-6-6-1000xsum-8epoche")
model = AutoModelForSeq2SeqLM.from_pretrained("NuppuCat/distillBart-6-6-1000xsum-8epoche")

def gensummary(ARTICLE):
    inputs = tokenizer(
        ARTICLE,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    
    # 生成摘要
    summary_ids = model.generate(
        **inputs,
        max_length=400,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    
    # 解码生成的摘要
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return generated_summary



@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")


    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        generated_summary = gensummary(text)
        return jsonify({"summary": generated_summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
