import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score
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
from ast import literal_eval
from scipy.special import expit





import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--b', type=int, default = 8)
parser.add_argument('--e', type=int, default = 20)
parser.add_argument('--l', type=int, default = 3*0.00001)
parser.add_argument('--s', type=int, default = 1)
parser.add_argument('--s', type=int, default = 1)


args = parser.parse_args()






DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_segment_length=512
lr = args.l
epoch = args.e
batch_size = args.b 
n_layer = 6
dir_ = "/workspace/legal/downstream/eurlex"
path = "/workspace/legal/models/student-10.pth"
seed = args.s
set_seed(seed)
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
print("DEVICE: " , DEVICE)
print("MODEL_NAME: " , MODEL_NAME)
print("max_segment_length: ", max_segment_length)
print("lr: ", lr)
print("epoch: ", epoch)
print("batch_size: ", batch_size)
print("n_layer: ", n_layer)
print("Eurlex")
print("Seed:", seed)
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



class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss






label_list = list(range(100))
num_labels = len(label_list)

def preprocess_function(examples):
    # Tokenize the texts
    batch = tokenizer(
        examples["text"],
        padding='max_length',
        max_length=max_segment_length,
        truncation=True,
    )
    batch["labels"] = [[1 if label in labels else 0 for label in label_list] for labels in examples["labels"]]

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


train_dataset = load_dataset("lex_glue", "eurlex", split="train")
eval_dataset = load_dataset("lex_glue", "eurlex" , split="validation")



train_dataset = train_dataset.map(
      preprocess_function,
      batched=True,
  )

eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
            )




def compute_metrics(p: EvalPrediction):
    # Fix gold labels
    y_true = np.zeros((p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32)
    y_true[:, :-1] = p.label_ids
    y_true[:, -1] = (np.sum(p.label_ids, axis=1) == 0).astype('int32')
    # Fix predictions
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = (expit(logits) > 0.5).astype('int32')
    y_pred = np.zeros((p.label_ids.shape[0], p.label_ids.shape[1] + 1), dtype=np.int32)
    y_pred[:, :-1] = preds
    y_pred[:, -1] = (np.sum(preds, axis=1) == 0).astype('int32')
    # Compute scores
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}




M1 = pred_model()
M2 = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels,num_hidden_layers = n_layer).to(DEVICE)

M2.bert.embeddings = M1.bert.bert.embeddings
M2.bert.encoder = M1.bert.bert.encoder





metric_name = "macro-f1"


args = TrainingArguments(
output_dir = dir_,
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
#push_to_hub=True,
)




trainer = MultilabelTrainer(
    model=M2,
    args= args,
    train_dataset=train_dataset ,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]


)

t = trainer.train()
e = trainer.evaluate()

trainer.log_metrics("train", t.metrics)
trainer.save_metrics("train", t.metrics)

trainer.log_metrics("eval", e)
trainer.save_metrics("eval", e)




