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
from keybert import KeyBERT

import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--b', type=int, default = 32)
parser.add_argument('--e', type=int, default = 1)
parser.add_argument('--cl', type=int, default = 512)
parser.add_argument('--l1', type=float, default = 0.00001)
parser.add_argument('--n', type=int, default = 100000)

args = parser.parse_args()

batch_size = args.b
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = args.e
CUTOFF_LEN = args.cl
l1 = args.l1
end = args.n



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
print("batch_size: ", batch_size)
print("MODEL_NAME: ", MODEL_NAME)
print("EPOCHS: ", EPOCHS)
print("CUTOFF_LEN: ", CUTOFF_LEN)
print("l1: ", l1)
print("Total Number of Samples: ",end)
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





kw_model = KeyBERT()


class DataClass(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)



def masking(a1, a2, value_to_replace=103):

    a1 = a1.numpy()
    a2 = a2.numpy()
    # Create a mask for non-zero values in a2
    mask = a2 != 0

    # Get indices where a2 has non-zero values
    indices = np.nonzero(mask)

    # Replace values in a1 based on indices from a2
    for i, j in zip(*indices):

        value = a2[i][j]

        if(a2[i][j] != 101 and a2[i][j] != 102):

            a1[i][a1[i] == value] = value_to_replace



def helper(arr,tokenizer):

  arr_ = []

  for i in arr:

      n = int(len(i.split())*0.15)
      a = kw_model.extract_keywords(i, keyphrase_ngram_range=(1, 1), stop_words=None,top_n = n)

      s = ""
      for j in a:

          s += j[0] + " "

      arr_.append(s)

  inputs_ = tokenizer(arr_, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

  return inputs_



def data_process(tokenizer,text,max_length=512,seed = 1,batch_size = 32):

  inputs = tokenizer(text, return_tensors='pt',max_length=512, truncation=True, padding='max_length')

  inputs['labels'] = inputs.input_ids.detach().clone()

  inputs_ = helper(text,tokenizer)

  masking(inputs['input_ids'], inputs_['input_ids'])


  dataset = DataClass(inputs)

  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,drop_last=True, shuffle=True)


  return loader



class Model(nn.Module):

    # Constructor class
    def __init__(self, num_hidden_layers = 6 , n_dim = 2048):
        super(Model, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,num_hidden_layers = num_hidden_layers)
        # self.bert.cls.predictions.decoder = nn.Linear(in_features=768, out_features=n_dim)
        self.final = nn.Linear(30522, 2048)


    def forward(self,input_ids, attention_mask, labels ):

        output = self.bert( input_ids = input_ids,
                                attention_mask = attention_mask,
                                labels = labels)
        
        x = self.final(output[1][:,0,:])

        return output[0],x


data = load_dataset("pile-of-law/pile-of-law",'all',split="train", streaming=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print()
print()
print("data-streaming object",data)
print()
print()



def student_gan_loss(student_logits,D):


    labels = torch.ones(len(student_logits), 1).to(DEVICE)
    g_loss = criterion(D(student_logits), labels)

    return g_loss


def kd_loss(teacher_logits,student_logits,T=2):

    soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
    soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)    #log_softmax

    val = kl_loss(soft_prob,soft_targets)

    return val


def cosine_sim_loss(teacher_logits,student_logits,T= 2):


    teacher_logits =  nn.functional.softmax(teacher_logits/ T, dim=-1)
    student_logits = nn.functional.softmax(student_logits/ T, dim=-1)

    y = torch.ones(student_logits.shape[0])
    y = y.to(DEVICE)

    res = cosine_loss(student_logits,teacher_logits,y)

    return res


cross_loss = nn.CrossEntropyLoss()
cosine_loss = nn.CosineEmbeddingLoss()
criterion = nn.BCELoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")



def train_epoch_distilled(train_d2,M1,M2,optimizer,scheduler,a = 0.33,b = 0.33 ,c = 0.33):



    M1 = M1.eval()  # Teacher set to evaluation mode
    M2 = M2.train()
    total_train_loss = 0
    total_entropy_loss_val = 0
    total_mask_loss = 0
    teacher_total_mask_loss = 0
    total_cosine_loss_val = 0


    steps = 0
    d1 = {}


    for batch in tqdm(train_d2):


        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        # D_optimizer.zero_grad()

        with torch.no_grad():

            outputs1 = M1(input_ids = input_ids,
                          attention_mask = attention_mask,
                          labels = labels)


        outputs2 = M2(input_ids = input_ids,
                      attention_mask = attention_mask,
                      labels = labels)


        teacher_logits = outputs1[1]
        student_logits = outputs2[1]



        entropy_loss_val =  kd_loss(teacher_logits,student_logits)

        auto_mask_loss = outputs2[0]
        teacher_auto_mask_loss = outputs1[0]

        cosine_loss_val = cosine_sim_loss(teacher_logits,student_logits)

        loss = (a*entropy_loss_val + b*auto_mask_loss + c*cosine_loss_val)


        total_train_loss += loss.item()
        total_entropy_loss_val += entropy_loss_val.item()
        total_mask_loss += auto_mask_loss.item()
        teacher_total_mask_loss += teacher_auto_mask_loss.item()
        total_cosine_loss_val += cosine_loss_val.item()
     

        steps += 1

        loss.backward()

        optimizer.step()

        scheduler.step()




    d1["train_loss"] = total_train_loss/steps
    d1["train_entropy_loss"] = total_entropy_loss_val/steps
    d1["train_mask_loss"] = total_mask_loss/steps
    d1["teacher_train_mask_loss"] = teacher_total_mask_loss/steps
    d1["train_cosine_loss"] = total_cosine_loss_val/steps

    return d1





def eval_model_distilled(val_d2,M1,M2,a = 0.33,b = 0.33 ,c = 0.33):



    M1 = M1.eval()  # Teacher set to evaluation mode
    M2 = M2.train()
    total_val_loss = 0
    total_entropy_loss_val = 0
    total_mask_loss = 0
    teacher_total_mask_loss = 0
    total_cosine_loss_val = 0

    steps = 0
    d1 = {}



    for batch in tqdm(val_d2):


        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)


        with torch.no_grad():

            outputs1 = M1(input_ids = input_ids,
                          attention_mask = attention_mask,
                          labels = labels)



            outputs2 = M2(input_ids = input_ids,
                          attention_mask = attention_mask,
                          labels = labels)


            teacher_logits = outputs1[1]
            student_logits = outputs2[1]


        


            entropy_loss_val =  kd_loss(teacher_logits,student_logits)


            auto_mask_loss = outputs2[0]
            teacher_auto_mask_loss = outputs1[0]



            cosine_loss_val = cosine_sim_loss(teacher_logits,student_logits)


            loss = (a*entropy_loss_val + b*auto_mask_loss + c*cosine_loss_val )

            total_val_loss  += loss.item()
            total_entropy_loss_val += entropy_loss_val.item()
            total_mask_loss += auto_mask_loss.item()
            teacher_total_mask_loss += teacher_auto_mask_loss.item()
            total_cosine_loss_val += cosine_loss_val.item()
     

            steps += 1

    d1["val_loss"] = total_val_loss/steps
    d1["val_entropy_loss"] = total_entropy_loss_val/steps
    d1["val_mask_loss"] =  total_mask_loss/steps
    d1["teacher_val_mask_loss"] =  teacher_total_mask_loss/steps
    d1["val_cosine_loss"] =  total_cosine_loss_val/steps


    return d1



def train_distilled(data,M1,M2,optimizer,scheduler):



  arr = []
  steps = 0

  train_loss = 0
  train_entropy_loss = 0
  train_mask_loss = 0
  train_cosine_loss = 0
  teacher_train_mask_loss = 0


  val_loss = 0
  val_entropy_loss = 0
  val_mask_loss = 0
  val_cosine_loss = 0
  teacher_val_mask_loss = 0



  results_ = defaultdict(list)
  results = defaultdict(int)


  for i,d in tqdm(enumerate(data)):

    arr.append(d["text"])

    if(len(arr) == 10000):

      steps += 1

      s = time.time()

      train_ssl_, val_ssl_ = train_test_split(arr,test_size=0.15)

      train_d2 = data_process(tokenizer,train_ssl_,seed = i)
      val_d2 = data_process(tokenizer,val_ssl_,seed = i)

      

      res1 = train_epoch_distilled(train_d2,M1,M2,optimizer,scheduler)


      res2 = eval_model_distilled(val_d2,M1,M2)

      e = time.time()


      train_loss += res1["train_loss"]
      train_entropy_loss +=  res1["train_entropy_loss"]
      train_mask_loss +=  res1["train_mask_loss"]
      teacher_train_mask_loss +=  res1["teacher_train_mask_loss"]
      train_cosine_loss +=  res1["train_cosine_loss"]
      



      val_loss +=  res2["val_loss"]
      val_entropy_loss +=  res2["val_entropy_loss"]
      val_mask_loss +=  res2["val_mask_loss"]
      teacher_val_mask_loss +=  res2["teacher_val_mask_loss"]
      val_cosine_loss +=  res2["val_cosine_loss"]



      for i_,j in res1.items():

        results_[i_].append(j)

        print((i_,j),end = " ")

      print()
      print()
      print()
      print()
      print()
      print()

      for i_,j in res2.items():

        results_[i_].append(j)

        print((i_,j),end = " ")


      print()
      print()
      print()
      print()
      print()
      print()



      print()
      print()
      print()
      print()
      print("<->"*100)
      print("<->"*100)
      print()
      print()
      print("Steps: ", steps)
      print("Time: ", (e-s))
      print()
      print()
      print("<->"*100)
      print("<->"*100)
      print()
      print()
      print()
      print()


      arr = []

    if(i+1== end):



      results["train_loss"] = [train_loss/steps]
      results["train_entropy_loss"] = [train_entropy_loss/steps]
      results["train_mask_loss"] = [train_mask_loss/steps]
      results["teacher_train_mask_loss"] = [teacher_train_mask_loss/steps]
      results["train_cosine_loss"] = [train_cosine_loss/steps]


      results["val_loss"] = [val_loss/steps]
      results["val_entropy_loss"] = [val_entropy_loss/steps]
      results["val_mask_loss"] = [val_mask_loss/steps]
      results["teacher_val_mask_loss"] = [teacher_val_mask_loss/steps]
      results["val_cosine_loss"] = [val_cosine_loss/steps]

      print()
      print()
      print()
      print()
      print("*"*100)
      print("*"*100)
      print()
      print()
      print()
      print()


      for i_,j in results.items():


        print((i_, j),end = " ")

      print()
      print()
      print()
      print()
      print("*"*100)
      print("*"*100)
      print()
      print()
      print()
      print()

      return results, results_









M1 = Model(12).to(DEVICE)
M2 = Model(6).to(DEVICE)



print()
print()
print()
print()
print(M1)
print()
print()
print()
print()
print()
print()
print()
print()
print(M2)
print()
print()
print()
print()



path1 = "/workspace/legal/results/"
path2 = "/workspace/legal/models/"




optimizer = AdamW(M2.parameters(), lr=l1, correct_bias=False, weight_decay=0.01)

total_steps = int((10000*0.85*30)/64) * EPOCHS
 
warmup_steps = int(total_steps * 0.1)  
 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)




if(EPOCHS > 1):

  student_ = "student-" + str(EPOCHS-1) + ".pth"


  checkpoint = torch.load(path2 + student_)
  M2.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])






for i in range(EPOCHS,EPOCHS+3):


  csv1 = "dis-kd-" + str(i) + ".csv"
  csv2 = "dis-kd-steps-" + str(i) + ".csv"

  student = "student-" + str(i) + ".pth"

  print()
  print("*")
  print()
  print()
  print()
  print()
  print()
  print()
  print()
  print()
  print("Epoch: ", i )
  print()
  print()
  print()
  print()
  print()
  print()
  print()
  print()
  print()
  print()
  print()
  print()
  print("*")
  print()




  res,res_  = train_distilled(data,M1,M2,optimizer,scheduler)

  df1 = pd.DataFrame(res)
  df2 = pd.DataFrame(res_)





  df1.to_csv(path1 + csv1 , index = False)
  df2.to_csv(path1 + csv2, index = False)



  torch.save({
  
        'model_state_dict': M2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
  
    }, path2 + student)



