# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:42:09 2023

@author: One
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:02:02 2023

@author: One
"""



# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

# Torch ML libraries
import transformers
from transformers import AutoModelForSequenceClassification,AutoConfig,AutoTokenizer
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Misc.
import warnings
warnings.filterwarnings('ignore')

# Set intial variables and constants
# %config InlineBackend.figure_format='retina'
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
# Graph Designs
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

# Random seed for reproducibilty
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = ['D:/GRAM/BERT/hatespeech_detection/Classified/fff.csv','D:/GRAM/BERT/hatespeech_detection/Classified/coronavirus.csv',
         'D:/GRAM/BERT/hatespeech_detection/Classified/fakenews.csv','D:/GRAM/BERT/hatespeech_detection/Classified/hatespeech.csv',
         'D:/GRAM/BERT/hatespeech_detection/Classified/immigrants.csv','D:/GRAM/BERT/hatespeech_detection/Classified/metoo.csv']
#%%
# def pltcount(path):
#     plt.figure()
#     df = pd.read_csv(path)
#     df.shape
#     df.head()
#     df.isnull().sum()
#     sns.countplot(df,x = 'class')
#     plt.xlabel('review class');

# for path in paths:
#     pltcount(path)

#%%
def getcombineddf(paths)    :
    custom_columns = ['text', 'class']
    cdf = pd.DataFrame(columns=custom_columns)
    for path in paths:
        df = pd.read_csv(path)
        cdf = pd.concat([cdf, df], ignore_index=True)
    return cdf
df = getcombineddf(paths)
plt.figure()
df.shape
df.head()
df.isnull().sum()
sns.countplot(df,x = 'class')
plt.xlabel('review class');
# 定义要删除的标签和删除比例
label_to_remove = 'neutral'
remove_ratio = 0.7  # 30%

# 设置随机种子
random_seed = 42

# 随机选择要删除的行，并设置随机种子
rows_to_remove = df[df['class'] == label_to_remove].sample(frac=remove_ratio, random_state=random_seed)

# 使用 ~ 操作符保留未被选择的行
df = df[~df.index.isin(rows_to_remove.index)]
##############################
# df = df.sample(n=100, random_state=42)
plt.figure()
df.shape
df.head()
df.isnull().sum()
sns.countplot(df,x = 'class')
plt.xlabel('review class');
plt.figure()
def to_sentiment(cname):
    
    if cname == 'neutral':
        return 0
    elif cname == 'offensive':
        return 1
    elif cname == 'positive':
        return 2
    elif cname == 'hatespeech':
        return 3
    elif cname == 'sexism':
        return 4
    

# Apply to the dataset 
df['label'] = df['class'].apply(to_sentiment)
# Plot the distribution
class_names = ['neutral', 'offensive', 'positive','hatespeech','sexism']
ax = sns.countplot(df,x = 'label')
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
#%%
MODEL_NAME = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def preprocess_function(examples):    
    return tokenizer(examples["text"], truncation=True, max_length=160)
from datasets import Dataset
ds_train = Dataset.from_pandas(df_train)
ds_val = Dataset.from_pandas(df_val)
ds_test = Dataset.from_pandas(df_test)
ds_train_tokenized = ds_train.map(preprocess_function, batched=True)
ds_val_tokenized = ds_val.map(preprocess_function, batched=True)
ds_test_tokenized = ds_test.map(preprocess_function, batched=True)
NUM_LABELS = len(class_names)

#%%
from peft import (
    get_peft_model, PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training
    
)

peft_config = LoraConfig(
r=8,
lora_alpha=16,
target_modules=["qurey", "value"],
lora_dropout=0.01,
task_type="SEQ_CLS",
modules_to_save=["classifier"],
# inference_mode=False,
)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model = get_peft_model( model,peft_config)
# model.classifier.original_module.out_features=len(class_names)
# model.classifier.modules_to_save.default.out_features=len(class_names)

# model.load_state_dict(torch.load('d:/GRAM/BERT/demo/best_model_state.bin'))
model = model.to(device)
# for param in model.bert.parameters():
#     param.requires_grad = False
# Number of hidden units
# print(bert_model.config.hidden_size)
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(model)
print(model)
#%%
from transformers import TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support
NUM_EPOCHS = 10
training_args = TrainingArguments(
    output_dir='feeeback-classifier',
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    report_to="none",
    evaluation_strategy="epoch",
    save_strategy="epoch",

)
import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train_tokenized,
    eval_dataset=ds_val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    #data_collator=data_collator,
)
trainer.train()
trainer.save_model("D:/GRAM/BERT/results/trained_LORA_model_state.bin")
#%%
log_history  = trainer.state.log_history
def getplt(klist):
    for k in klist:
        v = [entry.get(k, None) for entry in log_history]
        v = [loss for loss in v if loss is not None]
        plt.plot(v, label=k)
# losses = [entry.get('loss', None) for entry in log_history]
# losses = [loss for loss in losses if loss is not None]
# val_loss = [entry['eval_loss'] for entry in log_history]
# evlacc = [entry['eval_accuracy'] for entry in log_history]
# 绘制损失曲线
plt.figure(figsize=(10, 5))
# plt.plot(losses, label='trainLoss')
getplt(['eval_loss','eval_accuracy'])
# plt.plot(val_loss, label='valLoss')
# plt.plot(evlacc, label='Accuracy')
plt.title('eval Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%

# losses = [entry.get('loss', None) for entry in log_history]
# losses = [loss for loss in losses if loss is not None]
# val_loss = [entry['eval_loss'] for entry in log_history]
# evlacc = [entry['eval_accuracy'] for entry in log_history]
# 绘制损失曲线
plt.figure(figsize=(10, 5))
# plt.plot(losses, label='trainLoss')
getplt(['loss'])
# plt.plot(val_loss, label='valLoss')
# plt.plot(evlacc, label='Accuracy')
plt.title('train Loss Curve')
plt.xlabel('time')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%
# 存储真实标签和预测标签
true_labels = []
predicted_labels = []
model.eval()
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');
# 使用测试集进行推断
test_dataloader = DataLoader(ds_test_tokenized,  shuffle=True)
data = next(iter(test_dataloader))
print(data.keys())
with torch.no_grad():
    for batch in test_dataloader:
        
        input_ids=torch.tensor(batch['input_ids']).cuda().unsqueeze(0)
        attention_mask=torch.tensor(batch['attention_mask']).cuda().unsqueeze(0)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # print(outputs.logits)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels=batch['label']
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predictions)
# Run inference on the test set
##############################################
# with torch.no_grad():
#     for example in ds_test_tokenized:
#         # print(example)
#         input_ids = torch.tensor(example['input_ids']).cuda().unsqueeze(0)  # Add a batch dimension
#         attention_mask =torch.tensor( example['attention_mask']).cuda().unsqueeze(0)  # Add a batch dimension
#         label = example['label']

#         outputs = model(input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         predicted_label = torch.argmax(logits, dim=1).item()

#         true_labels.append(label)
#         predicted_labels.append(predicted_label)

# 计算混淆矩阵
#%%

cm = confusion_matrix(true_labels, predicted_labels)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)
# show_confusion_matrix(conf_matrix)
#%%
print(classification_report(true_labels, predicted_labels, target_names=class_names))
torch.save(model.state_dict(), 'D:/GRAM/BERT/hatespeech_detection/Classified/lora_bert_10_epoche_model_state.bin')
trainer.save_model("D:/GRAM/BERT/results/trained_LORA_model_10epoche_state.bin")