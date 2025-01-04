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


parser = argparse.ArgumentParser()



parser.add_argument('--e', type=int, default = 50)
parser.add_argument('--b', type=int, default = 8)
parser.add_argument('--l', type=float, default = 1*0.00001)
parser.add_argument('--f', type=int, default = 1)
parser.add_argument('--s', type=int, default = 1)

args = parser.parse_args()


EPOCHS = args.e
batch_size = args.b
lr = args.l
flag = args.f
seed = args.s






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

num_hidden_layers = 6
folder_name_ = f"/workspace/fin/downstream/fin-ner/"
path = "/workspace/fin/models/student-10.pth"





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
        [" ".join([token for token in sentence_tokens if token is not None]) for sentence_tokens in examples["tokens"]],
        truncation=True,
        is_split_into_words=False
    )
    all_labels = examples["tags"]
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
            for sentence_tokens in examples["tokens"]
        ],
        truncation=True,
        is_split_into_words=False
    )

    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs



train_dataset_  = load_dataset("gtfintechlab/finer-ord-bio",split = "train",trust_remote_code=True)
val_dataset_  = load_dataset("gtfintechlab/finer-ord-bio",split = "validation",trust_remote_code=True)
test_dataset_  = load_dataset("gtfintechlab/finer-ord-bio",split = "test",trust_remote_code=True)


train_dataset = train_dataset_.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=train_dataset_.column_names,
)

val_dataset = val_dataset_.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=val_dataset_.column_names,
)

test_dataset = test_dataset_.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=test_dataset_.column_names,
)



data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

metric = evaluate.load("seqeval")

label_names = ['O', 'PER_B' , 'PER_I' , 'LOC_B' , 'LOC_I', 'ORG_B' , 'ORG_I' ]

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
eval_dataloader = DataLoader(
    val_dataset, collate_fn=data_collator, batch_size=batch_size
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

def convert_labels(label):
    # Define a dictionary for the conversion
    label_mapping = {
        'LOC_B': 'B-LOC',
        'LOC_I': 'I-LOC',
        'PER_B': 'B-PER',
        'PER_I': 'I-PER',
        'ORG_B': 'B-ORG',
        'ORG_I': 'I-ORG',
        'O': 'O'  # 'O' stays the same as it's outside any entity
    }

    # Return the mapped label or the original if not found in the mapping
    return label_mapping.get(label, label)


def convert_labels_in_list(true_labels):
    # Apply the conversion to all labels in the list of lists
    return [[convert_labels(label) for label in sentence] for sentence in true_labels]



data = {
    'LOC': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(9)},
    'ORG': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(9)},
    'PER': {'precision': np.float64(0.0), 'recall': np.float64(0.0), 'f1-score': np.float64(0.0), 'support': np.int64(9)},
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

# Convert the data




def model_fit(model,train_dataloader,
              eval_dataloader,
              optimizer, lr_scheduler,
              max_grad_norm=1.0):


    model.train()
    train_epoch_loss = 0


    for step, batch in tqdm(enumerate(train_dataloader)):

        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss


        loss.backward()

        clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        lr_scheduler.step()

        train_epoch_loss += loss.item()



    train_avg_loss = train_epoch_loss / len(train_dataloader)



    model.eval()
    all_predictions = []
    all_labels = []

    for batch in tqdm(eval_dataloader):
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
    converted_labels = convert_labels_in_list(true_labels)
    converted_predictions = convert_labels_in_list(true_predictions)

    report = classification_report(converted_labels, converted_predictions,output_dict=True)

    report["train_avg_loss"] = train_avg_loss

    print()
    print("*"*100)
    print()
    print(report)
    print()
    print("*"*100)
    print()

    return convert_types(report)







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

    print()
    print("*"*100)
    print()
    print("epoch: ", e + 1 )
    print()
    print("*"*100)
    print()


    df = model_fit(M2,train_dataloader,
                test_dataloader,
                optimizer, scheduler,
    )

    with open(filename + str(e+1) + ".json", 'w') as json_file:
        json.dump(df, json_file, indent=4)

