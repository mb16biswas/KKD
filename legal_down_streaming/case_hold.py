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
    AutoModelForMultipleChoice

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
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union






import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--b', type=int, default = 8)
parser.add_argument('--e', type=int, default = 20)
parser.add_argument('--l', type=int, default = 3*0.00001)
parser.add_argument('--s', type=int, default = 1)

args = parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_segment_length=512
lr = args.l
epoch = args.e
batch_size = args.b 
dir_ = "/workspace/legal/downstream/case_hold"
path = "/workspace/legal/models/student-10.pth"
n_layer = 6 
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
print("Case-Hold")
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





@dataclass
class DataCollatorForMultipleChoice:
  
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch




def preprocess_function(examples):

  
    first_sentences = [[context] * 5 for context in examples["context"]]
    second_sentences = examples["endings"]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}


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





train_dataset = load_dataset('lex_glue', "case_hold", split = "train")
eval_dataset = load_dataset('lex_glue', "case_hold", split = "validation")




train_dataset = train_dataset.map(
      preprocess_function,
      batched=True,
  )

eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
            )




def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    # Compute macro and micro F1 for 5-class CaseHOLD task
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}




M1 = pred_model()
M2 = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased",num_hidden_layers = n_layer).to(DEVICE)
M2.bert.embeddings = M1.bert.bert.embeddings
M2.bert.encoder = M1.bert.bert.encoder






metric_name = "macro-f1"



training_args = TrainingArguments(
    
    output_dir = dir_,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch,
    weight_decay=0.01,
    save_total_limit=2,
    # load_best_model_at_end=True,
    metric_for_best_model=metric_name,

)

trainer = Trainer(
    model=M2,
    args=training_args,
    train_dataset= train_dataset ,
    eval_dataset= eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
)

trainer.train()



t = trainer.train()
e = trainer.evaluate()

trainer.log_metrics("train", t.metrics)
trainer.save_metrics("train", t.metrics)

trainer.log_metrics("eval", e)
trainer.save_metrics("eval", e)
