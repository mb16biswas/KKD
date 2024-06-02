import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import os
import transformers
from transformers import AutoModelForMaskedLM , AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset as D1, DataLoader
from tqdm import tqdm
from datasets import Dataset, load_dataset
from collections import defaultdict
import time
import argparse
from sklearn.metrics import f1_score,  r2_score, mean_squared_error
from sklearn.model_selection import KFold
parser = argparse.ArgumentParser()


parser.add_argument('--b', type=int, default = 64)  #8
parser.add_argument('--d', type=int, default = 1)
parser.add_argument('--e', type=int, default = 25)
parser.add_argument('--l', type=float, default = 3*0.00001)
parser.add_argument('--n', type=int, default = 10)
parser.add_argument('--cl', type=int, default = 64) #512

args = parser.parse_args()


batch_size = args.b
RANDOM_SEED = 16
MODEL_NAME = "ProsusAI/finbert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = args.e
CUTOFF_LEN = args.cl
n_layer = 6
l = args.l
n_splits = args.n
path = "/workspace/fin/models/student-10.pth"
dir_ = "/workspace/fin/downstream/fiqa/"




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
print("Regression: fiqa")
print("batch_size: ", batch_size)
print("RANDOM_SEED: ", RANDOM_SEED)
print("MODEL_NAME: ", MODEL_NAME)
print("DEVICE: ", DEVICE)
print("CUTOFF_LEN: ", CUTOFF_LEN)
print("n_layer: ", n_layer)
print("l: ", l)
print("n_splits: ", n_splits)
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


class Model(nn.Module):

    # Constructor class
    def __init__(self, num_hidden_layers = 6 , n_dim = 2048):
        super(Model, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,num_hidden_layers = num_hidden_layers)
        # self.bert.cls.predictions.decoder = nn.Linear(in_features=768, out_features=n_dim)
        self.final = nn.Linear(30522, n_dim)
  


    def forward(self,input_ids, attention_mask, labels ):

        output = self.bert( input_ids = input_ids,
                                attention_mask = attention_mask,
                                labels = labels)

        x = self.final(output[1][:,0,:])

        return x






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

  M1.final = nn.Linear(30522, 1)

  return M1.to(DEVICE)






tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(x,y):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        x,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding='max_length',
        return_tensors=None,
    )

    return {
            'review_text': x,
            'input_ids': result['input_ids'],
            'attention_mask': result['attention_mask'],
            'targets': torch.tensor(y, dtype=torch.float) ,
            "labels" : result["input_ids"].copy()
        }



def generate_and_tokenize_prompt(data_point):

    tokenized_full_prompt = tokenize(data_point['sentence'],data_point["score"])

    return tokenized_full_prompt


def create_data_pt(data,batch_size = batch_size):

    data.set_format(type='torch', columns=['sentence', 'input_ids', 'attention_mask', 'labels','targets'])
    data = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return data





def train_epoch(my_model, data_loader, optimizer,DEVICE, scheduler,loss_fn):

    my_model = my_model.train()
    total_mse = 0
    total_r2 = 0
    total_loss = 0
    steps = 0

    for d in tqdm(data_loader):

        input_ids = d["input_ids"].to(DEVICE)
        attention_mask = d["attention_mask"].to(DEVICE)
        targets = d["targets"].to(DEVICE)
        labels = d["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = my_model(input_ids=input_ids,
                           labels=labels,
                           attention_mask=attention_mask)

        outputs = outputs.view(-1)
        targets = targets.view(-1)

        loss = loss_fn(outputs, targets)

        t = targets.detach().cpu().numpy()
        p = outputs.detach().cpu().numpy()

        mse = mean_squared_error(t,p)
        r2 = r2_score(t,p)

        total_loss += loss.item()
        total_mse += mse
        total_r2  += r2

        steps += 1

        loss.backward()
        optimizer.step()
        scheduler.step()



    return (total_loss/steps, total_mse/steps , total_r2/steps )

def eval_model(my_model, data_loader,DEVICE,loss_fn):

    my_model = my_model.eval()
    total_mse = 0
    total_r2 = 0
    total_loss = 0
    steps = 0

    with torch.no_grad():

        for d in tqdm(data_loader):

            input_ids = d["input_ids"].to(DEVICE)
            attention_mask = d["attention_mask"].to(DEVICE)
            targets = d["targets"].to(DEVICE)
            labels = d["labels"].to(DEVICE)

            outputs = my_model(input_ids=input_ids,
                               labels=labels,
                              attention_mask=attention_mask)


            outputs = outputs.view(-1)
            targets = targets.view(-1)

            loss = loss_fn(outputs, targets)

            t = targets.detach().cpu().numpy()
            p = outputs.detach().cpu().numpy()

            mse = mean_squared_error(t,p)
            r2 = r2_score(t,p)

            total_loss += loss.item()
            total_mse += mse
            total_r2  += r2

            steps += 1

    return (total_loss/steps, total_mse/steps , total_r2/steps)




def train(model,train_d2,val_d2,DEVICE,EPOCHS,verbose = True):


    if(verbose == True):

        print()
        print()
        print()
        print(model)
        print()
        print()
        print()

    # model = model.to(DEVICE)

    tt = 0
    tt2 = 0

    d = defaultdict(list)

    optimizer = AdamW(model.parameters(), lr=l, correct_bias=False)

    total_steps = len(train_d2) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Set the loss function
    loss_fn = nn.MSELoss().to(DEVICE)

    # history = defaultdict(list)
    best_accuracy = 0


    for epoch in range(EPOCHS):

        print()
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 20)
        s = time.time()
        train_total_loss, train_total_mse,train_total_r2  = train_epoch(
        model,
        train_d2,
        optimizer,
        DEVICE,
        scheduler,
        loss_fn
        )
        e = time.time()
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
        print("Training time per Epoch: ",(e-s))
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


        tt += (e-s)

        s = time.time()
        val_total_loss, val_total_mse, val_total_r2  = eval_model(
        model,
        val_d2,
        DEVICE,
        loss_fn
        )
        e = time.time()
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
        print("Eval time per Epoch: ",(e-s))
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


        tt2 += (e-s)


        print()
        print()
        print()
        print()
        print(f' train_total_loss {train_total_loss:.5f}   <-> train_total_mse {train_total_mse:.5f}  <-> train_total_r2 {train_total_r2:.5f} ')
        print(f' val_total_loss {val_total_loss:.5f}   <-> val_total_mse {val_total_mse:.5f}  <-> val_total_r2 {val_total_r2:.5f} ')
        print()
        print()
        print()
        print()


        d["train_total_loss"].append(train_total_loss)
        d["train_total_mse"].append(train_total_mse)
        d["train_total_r2"].append(train_total_r2)
        d["val_total_loss"].append(val_total_loss)
        d["val_total_mse"].append(val_total_mse)
        d["val_total_r2"].append(val_total_r2)

    print()
    print()
    print("*"*100)
    print()
    print()
    print()
    print()
    print()
    print("*"*100)
    print()
    print()
    print()
    print("Total time taken train : ", tt)
    print("Total time taken eval : ", tt2)
    print("Average time per epoch train : ", tt/EPOCHS)
    print("Average time per epoch eval : ", tt2/EPOCHS)
    print()
    print()
    print()
    print("*"*100)
    print()
    print()
    print()
    print()
    print("*"*100)
    print()
    print()
    print()


    return d



# train_ssl = load_dataset("ChanceFocus/fiqa-sentiment-classification", split = "train[:10]")
# val_ssl = load_dataset("ChanceFocus/fiqa-sentiment-classification", split = "valid[:10]")


df = pd.read_csv("/workspace/data/Momojit/gan-kd/fin-bert/Data/reg.csv")


X = list(df["sentence"])
y = list(df["score"])

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
              "score" : list(train_fold["score"])})

    df_val = pd.DataFrame({"sentence" : list(val_fold["sentence"]),
              "score" : list(val_fold["score"])})

    train_ssl = Dataset.from_pandas(df_train)
    val_ssl = Dataset.from_pandas(df_val)


    train_d = train_ssl.shuffle().map(generate_and_tokenize_prompt)
    val_d = val_ssl.shuffle().map(generate_and_tokenize_prompt)

    train_d2 =  create_data_pt(train_d,batch_size)
    val_d2 =  create_data_pt(train_d,batch_size)



    M1 = pred_model()
    history1 = train(M1,train_d2,val_d2, DEVICE,EPOCHS)
    d = pd.DataFrame(history1)
    path_csv = dir_ + "res" + str(cnt) + ".csv"
    d.to_csv(path_csv, index = False)
    del M1
    torch.cuda.empty_cache()

    cnt += 1
