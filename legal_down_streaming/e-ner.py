from datasets import load_dataset
from transformers import AutoTokenizer
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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification , AutoModelForTokenClassification
import pandas as pd
from ast import literal_eval
from scipy.special import expit
import torch
from transformers import AutoModelForQuestionAnswering
import evaluate
from tqdm import tqdm
import collections
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import json
import os
from torch.nn.utils import clip_grad_norm_
import argparse
from sklearn.model_selection import train_test_split
import ast

parser = argparse.ArgumentParser()



parser.add_argument('--e', type=int, default = 100)
parser.add_argument('--b', type=int, default = 8)
parser.add_argument('--l', type=float, default = 3*0.00001)
parser.add_argument('--f', type=int, default = 1)
parser.add_argument('--s', type=int, default = 1)

args = parser.parse_args()


EPOCHS = args.e
batch_size = args.b
lr = args.l
flag = args.f
seed = args.s
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
folder_name_ = f"/workspace/legal/downstream/legal-ner/"
path = "/workspace/legal/models/student-10.pth"




os.makedirs(folder_name_, exist_ok=True)
set_seed(seed)

print("Configuration Details:")
print(f"-----------------------")
print(f"device: {device}")
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"tokenizer: {tokenizer}")
print(f"batch_size: {batch_size}")
print(f"learning rate (lr): {lr}")
print(f"EPOCHS: {EPOCHS}")
print(f"num_hidden_layers: {num_hidden_layers}")
print(f"folder_name_: {folder_name_}")




def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels




def tokenize_and_align_labels(examples):
    # Iterate through each sentence's tokens and join them into a string
    tokenized_inputs = tokenizer(
        [" ".join([token for token in sentence_tokens if token is not None]) for sentence_tokens in examples["words"]],
        truncation=True,
        is_split_into_words=False
    )
    all_labels = examples["ner"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        # Check if word_id is within the valid range of labels
        if word_id is not None and word_id < len(labels):
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        else:
            # Handle cases where word_id is out of range
            new_labels.append(-100)  # Assign -100 for out-of-range word_ids

    return new_labels



def tokenize_and_align_labels(examples):


    tokenized_inputs = tokenizer(
        [
            # Check for None in sentence_tokens and its individual tokens
            " ".join([token if token is not None else '' for token in sentence_tokens])
            if sentence_tokens is not None else ''  # If sentence_tokens itself is None, return an empty string
            for sentence_tokens in examples["words"]
        ],
        truncation=True,
        is_split_into_words=False
    )

    all_labels = examples["ner"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


ner_to_int = {
 'O': 0,
 'B-LOC': 1,
 'I-LOC': 2,
 'B-ORG': 3,
 'I-ORG': 4,
 'B-PER': 5,
 'I-PER': 6,
 'B-MISC': 7,
 'I-MISC': 8,

}





def convert_ner_to_int(example):
    # Convert the 'ner' labels from string to integer using the map
    example['ner'] = [ner_to_int[label] for label in example['ner']]
    return example



df = pd.read_csv("/workspace/legal/downstream/legal-ner/data/edger_ner_data_all-final-v4.csv")

df["ner"] = df["ner"].apply(ast.literal_eval)
df["words"] = df["words"].apply(ast.literal_eval)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset_ = Dataset.from_pandas(train_df)
test_dataset_ = Dataset.from_pandas(test_df)

train_dataset_ = train_dataset_.map(convert_ner_to_int)
test_dataset_ = test_dataset_.map(convert_ner_to_int)

train_dataset = train_dataset_.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=train_dataset_.column_names,
)



test_dataset = test_dataset_.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=test_dataset_.column_names,
)



data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

metric = evaluate.load("seqeval")

label_names = ['O',
 'B-LOC',
 'I-LOC',
 'B-ORG',
 'I-ORG',
 'B-PER',
 'I-PER',
 'B-MISC',
 'I-MISC']



id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }




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

  M1 = Model()

  print()
  print("*"*100)
  print()
  print()
  print(M1)

  print()
  print()
  print("*"*100)
  print()

  checkpoint = torch.load(path)
  M1.load_state_dict(checkpoint['model_state_dict'],strict=False)

  return M1.to(device)




M1 = pred_model()
M2 = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    num_hidden_layers = num_hidden_layers,
    ignore_mismatched_sizes=True
).to(device)


M2.bert.embeddings = M1.bert.bert.embeddings
M2.bert.encoder = M1.bert.bert.encoder



train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)

test_dataloader = DataLoader(
    test_dataset, collate_fn=data_collator, batch_size=batch_size
)






def postprocess(predictions, labels):
    # predictions = predictions.detach().cpu().clone().numpy()
    # labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions




data = {
    'LOC': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(9)},
    'ORG': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(9)},
    'PER': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(9)},
    'MISC': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(9)},
    'micro avg': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(27)},
    'macro avg': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(27)},
    'weighted avg': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(27)}
}


# Function to recursively convert numpy types to native Python types
def convert_types(d):
    if isinstance(d, dict):
        return {key: convert_types(value) for key, value in d.items()}
    elif isinstance(d, (np.int64, np.float64)):  # Check for numpy.int64 or numpy.float64
        return float(d) if isinstance(d, np.float64) else int(d)  # Convert to Python float or int
    else:
        return d




def model_fit(model,train_dataloader,
              test_dataloader ,
              optimizer, lr_scheduler):


    model.train()
    train_epoch_loss = 0
    all_predictions = []
    all_labels = []

    for batch in tqdm(train_dataloader):

        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Collect predictions and labels for metric calculation
        all_predictions.extend(predictions.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())


        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_epoch_loss += loss.item()



    train_avg_loss = train_epoch_loss / len(train_dataloader)
    true_predictions, true_labels = postprocess(all_predictions, all_labels)
    true_predictions, true_labels = postprocess(all_predictions, all_labels)


    report = classification_report(true_labels, true_predictions,output_dict=True,zero_division = 0 )




    model.eval()
    all_predictions = []
    all_labels = []

    for batch in tqdm(test_dataloader):
        # Move batch data to device

        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Collect predictions and labels for metric calculation
        all_predictions.extend(predictions.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())




    # Compute metrics
    true_predictions, true_labels = postprocess(all_predictions, all_labels)
    true_predictions, true_labels = postprocess(all_predictions, all_labels)


    report2 = classification_report(true_labels, true_predictions,output_dict=True,zero_division = 0 )

    report = convert_types(report)
    report2 = convert_types(report2)

    merged_report = {}
    for key, value in report.items():
        merged_report[f"train_{key}"] = value
    for key, value in report2.items():
        merged_report[f"val_{key}"] = value


    merged_report["train_avg_loss"] = train_avg_loss


    return merged_report








optimizer = AdamW(M2.parameters(), lr=lr, weight_decay=0.01)

steps_per_epoch = len(train_dataloader)

total_training_steps = steps_per_epoch * EPOCHS

warmup_steps = int(0.05 * total_training_steps)


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)


filename = folder_name_ + "result-epoch-"
for e in range(EPOCHS):

    df = model_fit(M2,train_dataloader,
                test_dataloader,
                optimizer, scheduler,
    )

    with open(filename + str(e+1) + ".json", 'w') as json_file:
        json.dump(df, json_file, indent=4)


