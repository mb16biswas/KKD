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
parser.add_argument('--ms', type=int, default = 32)
parser.add_argument('--l', type=int, default = 3*0.00001)
parser.add_argument('--s', type=int, default = 1)

args = parser.parse_args()






DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_segments= args.ms
max_segment_length=128
lr = args.l
epoch = args.e
batch_size = args.b 
n_layer = 6 
dir_ = "/workspace/legal/downstream/ecthr_a"
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
print("max_segments: ", max_segments)
print("max_segment_length: ", max_segment_length)
print("lr: ", lr)
print("epoch: ", epoch)
print("batch_size: ", batch_size)
print("n_layer: ", n_layer)
print("Ecthr-a")
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



label_list = list(range(10))
num_labels = len(label_list)



@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class HierarchicalBert(nn.Module):

    def __init__(self, encoder, max_segments=64, max_segment_length=128):
        super(HierarchicalBert, self).__init__()
        supported_models = ['bert', 'roberta', 'deberta']
        assert encoder.config.model_type in supported_models  # other model types are not supported so far
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(max_segments + 1, encoder.config.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(max_segments + 1, encoder.config.hidden_size))
        # Init segment-wise transformer-based encoder
        self.seg_encoder = nn.Transformer(d_model=encoder.config.hidden_size,
                                          nhead=encoder.config.num_attention_heads,
                                          batch_first=True, dim_feedforward=encoder.config.intermediate_size,
                                          activation=encoder.config.hidden_act,
                                          dropout=encoder.config.hidden_dropout_prob,
                                          layer_norm_eps=encoder.config.layer_norm_eps,
                                          num_encoder_layers=2, num_decoder_layers=0).encoder

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):



        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
        else:
            token_type_ids_reshape = None


        encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                       attention_mask=attention_mask_reshape,
                                       token_type_ids=token_type_ids_reshape)[0]


        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        encoder_outputs = encoder_outputs[:, :, 0]

        seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)

        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask

        encoder_outputs += self.seg_pos_embeddings(seg_positions)

        seg_encoder_outputs = self.seg_encoder(encoder_outputs)

        outputs, _ = torch.max(seg_encoder_outputs, 1)

        return  SimpleOutput(last_hidden_state=outputs, hidden_states=outputs)





class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss






def preprocess_function(examples):


  case_template = [[0] * max_segment_length]
  batch = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
  for doc in examples['text']:

      doc_encodings = tokenizer(doc[:max_segments], padding='max_length',
                                  max_length= max_segment_length, truncation=True)
      batch['input_ids'].append(doc_encodings['input_ids'] + case_template * (
                  max_segments - len(doc_encodings['input_ids'])))
      batch['attention_mask'].append(doc_encodings['attention_mask'] + case_template * (
                  max_segments - len(doc_encodings['attention_mask'])))
      batch['token_type_ids'].append(doc_encodings['token_type_ids'] + case_template * (
                  max_segments - len(doc_encodings['token_type_ids'])))


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



train_dataset = load_dataset("lex_glue", "ecthr_a",split="train")
eval_dataset = load_dataset("lex_glue", "ecthr_a" ,split="validation")




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


M2.bert = HierarchicalBert(encoder=M2.bert, max_segments=max_segments, max_segment_length=max_segment_length).to(DEVICE)



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
