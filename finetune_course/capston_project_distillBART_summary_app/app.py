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
import time


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("NuppuCat/distillBart-6-6-1000xsum-8epoche")
model = AutoModelForSeq2SeqLM.from_pretrained("NuppuCat/distillBart-6-6-1000xsum-8epoche")
teacher = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
student_model_original = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6")
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

def gensummary2(ARTICLE):
    inputs = tokenizer(
        ARTICLE,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    
    # 生成摘要
    summary_ids = teacher.generate(
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

def gensummary3(ARTICLE):
    inputs = tokenizer(
        ARTICLE,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    
    # 生成摘要
    summary_ids = student_model_original.generate(
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

# function to calculate the model size
def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # 计算参数占用
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # 计算缓冲区占用
    total_size = param_size + buffer_size  # 总大小（字节）
    return total_size / (1024 ** 2)  # 转换为 MB
modelsize = get_model_size(model)
teachersieze = get_model_size(teacher)
studentoriginalsize  = get_model_size(student_model_original)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text")


    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        start_time = time.time()
        generated_summary = gensummary(text)
        end_time = time.time()
        runtime= end_time - start_time
        return jsonify({"summary": generated_summary, "runtime": runtime, "modelsize":modelsize})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/analyze2", methods=["POST"])
def analyze2():
    data = request.json
    text = data.get("text")


    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        start_time = time.time()
        generated_summary = gensummary2(text)
        end_time = time.time()
        runtime= end_time - start_time
        return jsonify({"summary": generated_summary,"runtime": runtime, "modelsize":teachersieze})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/analyze3", methods=["POST"])
def analyze3():
    data = request.json
    text = data.get("text")


    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        start_time = time.time()
        generated_summary = gensummary3(text)
        end_time = time.time()
        runtime= end_time - start_time
        return jsonify({"summary": generated_summary,"runtime": runtime, "modelsize":studentoriginalsize})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
