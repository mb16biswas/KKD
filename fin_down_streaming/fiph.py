import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from torch import nn
import glob
import shutil
from transformers import AutoModelForMaskedLM , AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import transformers
from transformers import (
    Trainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,

)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np
from torch import nn
from transformers.file_utils import ModelOutput
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import KFold



import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--b', type=int, default = 64)
parser.add_argument('--e', type=int, default = 25)
parser.add_argument('--l', type=float, default = 2*0.00001)
parser.add_argument('--n', type=int, default = 10)
parser.add_argument('--t', type=int, default = 1)



args = parser.parse_args()






DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_segment_length=64
lr = args.l
epoch = args.e
batch_size = args.b
n_layer = 6
n_splits = args.n
task = args.t
path = "/workspace/fin/models/student-10.pth"
print()
print()
print()
print()
print()
print()
print()
print()
print("*"*100)
print()
print()
print()
print()
print()
print()
print()
print()
print("Fiph")
print("DEVICE: " , DEVICE)
print("MODEL_NAME: " , MODEL_NAME)
print("max_segment_length: ", max_segment_length)
print("lr: ", lr)
print("epoch: ", epoch)
print("batch_size: ", batch_size)
print("n_layer: ", n_layer)
print("n_splits: ", n_splits)
print("task: ", task)
print()
print()
print()
print()
print()
print()
print()
print()
print("*"*100)
print()
print()
print()
print()
print()
print()
print()
print()


if(task == 1):

    dir_ = "/workspace/fin/downstream/fiph/"


else:

    dir_ = "/workspace/fin/downstream/fiph2/"




label_list = list(range(3))
num_labels = len(label_list)



def preprocess_function(examples):
    # Tokenize the texts
    batch = tokenizer(
        examples['sentence'],
        padding='max_length',
        max_length=max_segment_length,
        truncation=True,
    )
    batch["label"] = [label_list.index(label) for label in examples["label"]]

    return batch






class Model(nn.Module):

    # Constructor class
    def __init__(self, num_hidden_layers = 6 , n_dim = 1024):
        super(Model, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,num_hidden_layers = num_hidden_layers)
        # self.bert.cls.predictions.decoder = nn.Linear(in_features=768, out_features=n_dim)


    def forward(self,input_ids, attention_mask, labels ):

        output = self.bert( input_ids = input_ids,
                                attention_mask = attention_mask,
                                labels = labels)

        return output[0],output[1][:,0,:]



def pred_model():


  M1 = Model(n_layer)

  print()
  print("*"*100)
  print()
  print()
  print(M1)
  print("Path: ", path)
  print("n_layer: ", n_layer)
  print()
  print()
  print("*"*100)
  print()

  checkpoint = torch.load(path)
  M1.load_state_dict(checkpoint['model_state_dict'],strict=False)

  return M1.to(DEVICE)






def compute_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1, 'accuracy' : accuracy}


if(task == 1):

  df = pd.read_csv("/workspace/fin/data/100-percent.csv")

else:

  df = pd.read_csv("/workspace/fin/data/all-data.csv")


print()
print()
print("Len: ", len(df))
print()
print()

X = list(df["sentence"])
y = list(df["label"])

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cnt = 1


for train_, val_ in kf.split(X, y):

    print()
    print()
    print()
    print("*"*100)
    print("*"*100)
    print()
    print()
    print()
    print(f'Fold:{cnt}, Train set: {len(train_)}, val set:{len(val_)}')
    print()
    print()
    print()
    print("*"*100)
    print("*"*100)
    print()
    print()
    print()

    train_fold = df.iloc[train_]
    val_fold =  df.iloc[val_]

    df_train = pd.DataFrame({"sentence" : list(train_fold["sentence"]),
              "label" : list(train_fold["label"])})

    df_val = pd.DataFrame({"sentence" : list(val_fold["sentence"]),
              "label" : list(val_fold["label"])})

    train_ssl = Dataset.from_pandas(df_train)
    val_ssl = Dataset.from_pandas(df_val)

    train_dataset = train_ssl.map(
      preprocess_function,
      batched=True,
    )

    eval_dataset = val_ssl.map(
                preprocess_function,
                batched=True,
            )  


    M1 = pred_model(method)
    M2 = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels,num_hidden_layers = n_layer).to(DEVICE)

    M2.bert.embeddings = M1.bert.bert.embeddings
    M2.bert.encoder = M1.bert.bert.encoder



    metric_name = "macro-f1"

    final_dr = dir_ + str(cnt) + "/"

    args = TrainingArguments(
    output_dir = final_dr,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,

    )



    trainer = Trainer(
        M2,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )


    t = trainer.train()
    e = trainer.evaluate()

    trainer.log_metrics("train", t.metrics)
    trainer.save_metrics("train", t.metrics)

    trainer.log_metrics("eval", e)
    trainer.save_metrics("eval", e)

    del M1
    del M2
    torch.cuda.empty_cache()

    cnt += 1 
    


