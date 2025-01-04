
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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
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
import json
import argparse
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser()



parser.add_argument('--e', type=int, default = 10)
parser.add_argument('--t', type=int, default = 10)
parser.add_argument('--b', type=int, default = 16)
parser.add_argument('--l', type=float, default = 3*0.00001)
parser.add_argument('--f', type=int, default = 1)
parser.add_argument('--s', type=int, default = 1)

args = parser.parse_args()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
num_hidden_layers = 6
folder_name_ = f"/workspace/legal/downstream/policy-qa/"
folder_name_model = f"/workspace/legal/downstream/policy-qa/model/"
path = "/workspace/legal/models/student-10.pth"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
max_length = 512
stride = 128
n_best = 20
max_answer_length = 30
lr = args.l
batch_size = args.b
EPOCHS = args.e
Total = args.t
seed = args.s


set_seed(seed)


os.makedirs(folder_name_, exist_ok=True)
os.makedirs(folder_name_model, exist_ok=True)


print("device:", device)
print("MODEL_NAME:", MODEL_NAME)
print("tokenizer:", tokenizer)
print("num_hidden_layers:", num_hidden_layers)
print("folder_name_:", folder_name_)
print("max_length:", max_length)
print("stride:", stride)
print("n_best:", n_best)
print("max_answer_length:", max_answer_length)
print("lr:", lr)
print("EPOCHS:", EPOCHS)
print("Total:", Total)
print("batch_size:", batch_size)





def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs




def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        # Skip samples with no answers
        if not answer["answer_start"]:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs




def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs




metric2 = evaluate.load("squad_v2")
metric = evaluate.load("squad")



def update_predictions(predictions):
    for prediction in predictions:
        # If 'prediction_text' is an empty string, replace it with "."
        if prediction['prediction_text'] == "":
            prediction['prediction_text'] = '.'

        # Add 'no_answer_probability' key with value 0
        prediction['no_answer_probability'] = 0

    return predictions


def update_answers(answers_list):
    for answer in answers_list:
        # Check if 'answers["text"]' is an empty list
        if not answer['answers']['text']:
            answer['answers']['text'] = ['.']  # Replace with ['.']

    return answers_list


def compute_metrics2(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    # predicted_answers = convert_predictions_to_predictions2(predicted_answers)
    # theoretical_answers = convert_references_to_references2(theoretical_answers)

    predicted_answers = update_predictions(predicted_answers)
    theoretical_answers = update_answers(theoretical_answers)

    return metric2.compute(predictions=predicted_answers, references=theoretical_answers)





def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)





train_dataset_ = load_dataset("alzoubi36/policy_qa", split = "train",trust_remote_code=True)
validation_dataset_ = load_dataset("alzoubi36/policy_qa", split = "validation",trust_remote_code=True)
test_dataset_ = load_dataset("alzoubi36/policy_qa", split = "test",trust_remote_code=True)


train_dataset = train_dataset_.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=train_dataset_.column_names,
)

validation_dataset = validation_dataset_.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=validation_dataset_.column_names,
)

test_dataset = test_dataset_.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=test_dataset_.column_names,
)



train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")
test_set = test_dataset.remove_columns(["example_id", "offset_mapping"])
test_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    collate_fn=default_data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=batch_size
)

test_dataloader = DataLoader(
    test_set, collate_fn=default_data_collator, batch_size=batch_size
)



def model_fit2(model,train_dataloader,
              eval_dataloader,validation_dataset,
              validation_dataset_,
              test_dataloader,test_dataset,
              test_dataset_,
              optimizer, lr_scheduler,
              max_grad_norm = 1.0
              ):


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
    val_epoch_loss = 0
    start_logits = []
    end_logits = []

    for batch in tqdm(eval_dataloader):
        with torch.no_grad():

            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)



            start_logits.append(outputs.start_logits.cpu().detach().numpy())
            end_logits.append(outputs.end_logits.cpu().detach().numpy())



    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    val_metrics = compute_metrics(start_logits, end_logits, validation_dataset,validation_dataset_ )

    print()
    print("*"*100)
    print()


    val_metrics["train_avg_loss"] = train_avg_loss

    print()
    print("*"*100)
    print("val")
    print(val_metrics)
    print()
    print("*"*100)
    print()



    model.eval()
    start_logits_test = []
    end_logits_test = []

    for batch in tqdm(test_dataloader):
        with torch.no_grad():

            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)


            start_logits_test.append(outputs.start_logits.cpu().detach().numpy())
            end_logits_test.append(outputs.end_logits.cpu().detach().numpy())


    start_logits_test = np.concatenate(start_logits_test)
    end_logits_test = np.concatenate(end_logits_test)


    test_metrics = compute_metrics(start_logits_test, end_logits_test, test_dataset, test_dataset_)

    print()
    print("*"*100)
    print("test")
    print(test_metrics)
    print()
    print("*"*100)
    print()


    test_metrics_renamed = {
        f"test_{key}": value
        for key, value in test_metrics.items()
    }


    val_metrics_renamed = {
        f"val_{key}": value
        for key, value in val_metrics.items()
    }


    merged_metrics = {**val_metrics_renamed, **test_metrics_renamed}


    print()
    print("*" * 100)
    print("Merged Metrics:")
    print(merged_metrics)
    print()
    print("*" * 100)

    return merged_metrics



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
M2 = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, num_hidden_layers =  num_hidden_layers).to(device)

M2.bert.embeddings = M1.bert.bert.embeddings
M2.bert.encoder = M1.bert.bert.encoder


optimizer = AdamW(M2.parameters(), lr=lr, weight_decay=0.01)

steps_per_epoch = len(train_dataloader)

total_training_steps = steps_per_epoch * EPOCHS

warmup_steps = int(0.05 * total_training_steps)


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)




if(EPOCHS  > 1):

  student_ = "model-" + str(EPOCHS-1) + ".pth"


  checkpoint = torch.load(folder_name_model + student_)
  M2.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])









filename = folder_name_ + "result-epoch-"
for e in range(EPOCHS,EPOCHS+Total):


    student = "model-" + str(e) + ".pth"

    print()
    print("*"*100)
    print()
    print("epoch: ", e  )
    print()
    print("*"*100)
    print()


    df = model_fit2(M2,train_dataloader,
                eval_dataloader,validation_dataset,
                validation_dataset_,
                test_dataloader,test_dataset,
                test_dataset_,
                optimizer, scheduler,
                )


    with open(filename + str(e) + ".json", 'w') as json_file:
        json.dump(df, json_file, indent=4)



    torch.save({

        'model_state_dict': M2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),

    }, folder_name_model  + student)

