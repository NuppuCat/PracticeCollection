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
# 加载模型和 tokenizer
# Custom 模型：DistilBERT
custom_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
custom_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)
custom_sentiment = pipeline("sentiment-analysis", model=custom_model, tokenizer=custom_tokenizer)

# Llama3 模型（假设为 llama 1B instrct）
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")  # 替换成你需要的 Llama3 模型
llama_model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", num_labels=2)
llama_sentiment = pipeline("sentiment-analysis", model=llama_model, tokenizer=llama_tokenizer)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    model_name = data.get("model", "custom")  # 默认使用 custom 模型

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # 根据模型名称选择对应模型
        if model_name == "custom":
            result = custom_sentiment(text)
        elif model_name == "llama3":
            result = llama_sentiment(text)
        else:
            return jsonify({"error": "Invalid model selection"}), 400

        # 返回结果
        sentiment = result[0]["label"]
        score = result[0]["score"]
        return jsonify({"sentiment": sentiment, "score": score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
