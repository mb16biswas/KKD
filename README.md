# Keyword Knowledge Distilation (KKD)



## Instructions to run the code

### Required directory structure to save the results:

```
+-- workspace/
  |    fin/
  |   +-- data/
  |   +-- results/
  |   |   +--dis-kd-1.csv
  |   |   +--dis-kd-steps-1.csv
  |   |   ...
  |   +-- models/
  |   |   +--student-1.pth
  |   |   ...
  |   +-- downstream/
  |   |   +--fiph/
  |   |   ...
  |   |   ...
  |    legal/
  |   +-- data/
  |   +-- results/
  |   |   +--dis-kd-1.csv
  |   |   +--dis-kd-steps-1.csv
  |   |   ...
  |   +-- models/
  |   |   +--student-1.pth
  |   |   ...
  |   +-- downstream/
  |   |   +--ecthr_a/
  |   |   +--ecthr_b/
  |   |   ...

```

### Install the required packages:

```
pip install --upgrade pip
pip install torch==2.0.1
pip install datasets==2.16.1
pip install scikit-learn numpy pandas
pip install transformers==4.30
pip install -q -U trl accelerate git+https://github.com/huggingface/peft.git
pip install -q bitsandbytes einops sentencepiece
pip install evaluate seqeval
pip uninstall -y apex
pip install keybert

```
### Run the commands for KKD training

```
python fin_kkd_student.py
python legal_kkd_student.py
```
Available arguments:
- `--b`: Batch size. Default = 32
- `--e`: Current number of epoch . Default = 1
- `--cl`: Input sequence length of the Teacher and Student model . Default = 512
- `--l1`: Learning rate for training. Default = 0.00001
- `--n`: Total no of samples in the dataset. Default = 1200000



### Run the commands for the downstreaming tasks for Legal Domain

```
python legal_down_streaming/ecthr_a.py
python legal_down_streaming/ecthr_b.py
python legal_down_streaming/scotus.py
python legal_down_streaming/ledger.py
python legal_down_streaming/unfair-tos.py
python legal_down_streaming/eurlex.py
python legal_down_streaming/case_hold.py

```

Available arguments for ecthr_a, ecthr_b and scotus :

- `--b`: Batch size. Default = 8
- `--e`: Current number of epoch . Default = 20
- `--ms`: Paragraph Length . Default = 32
- `--l`: Learning rate for training. Default = 0.00003


Available arguments for ledger, unfair-tos, eurlex and case_hold :

- `--b`: Batch size. Default = 8
- `--e`: Current number of epoch . Default = 20
- `--l`: Learning rate for training. Default = 0.00003




### Run the commands for the downstreaming tasks for Finance Domain

```
python fin_down_streaming/fiqa.py
python fin_down_streaming/fiph.py 

```




Available arguments for fiph :

- `--b`: Batch size. Default = 64
- `--e`: Current number of epoch . Default = 25
- `--l`: Learning rate for training. Default = 2*0.00001
- `--n`: Number of K-fold. Default = 10
- `--t`: 1 := 100-percent, else := ALL-DATA 



Available arguments for fiqa :

- `--b`: Batch size. Default = 64
- `--e`: Current number of epoch . Default = 25
- `--l`: Learning rate for training. Default = 2*0.00001
- `--n`: Number of K-fold. Default = 10
