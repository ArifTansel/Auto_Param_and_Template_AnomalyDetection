import numpy as np
import pandas as pd
from datetime import datetime , timedelta
from sklearn.cluster import KMeans
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
import numpy as np
# Drain3 persistence
persistence = FilePersistence("drain3_state.json")
template_miner = TemplateMiner(persistence)

records = []

with open("HDFS.log", "r") as f:
    for idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        result = template_miner.add_log_message(line)
        records.append({"line_id": idx, "log": line, "cluster_id": result["cluster_id"]})

df = pd.DataFrame(records)

# Final template map
cluster_map = {c.cluster_id: c.get_template() for c in template_miner.drain.clusters}
for r in records:
    r["template"] = cluster_map.get(r["cluster_id"], None)

df = pd.DataFrame(records)
df.drop(columns=["cluster_id"], inplace=True)
# Parametreleri çıkar

df_unique_examples = df.groupby("template").first().reset_index()
for cluster in template_miner.drain.clusters:
    print(cluster)

"""# Extract Params With LLM

Çalıştırabilmek için GEMINI_API_KEY bulundurmak gerekir
"""

from tqdm import tqdm
prompt = """
Prompt:
You are an AI agent that extracts parameters from log templates.
I will provide you with:
A raw log line.
Its corresponding log template.
The log template contains <*> placeholders (masks). Your task is to return an array of parameter types in the same order that the <*> appear.
Allowed parameter types (keys):
USER
DATETIME
IP
DURATION
BLKNO
Example:
Log: 081111 092742 25457 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-5952182400333241163 terminating
Template: <*> <*> <*> dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_<*> terminating
Output: {{DATETIME,*,*,BLKNO}}
Instructions:

Only use the keys listed above. If paramater does not match any key write * in its place

Before Giving output always make a analyse for each "<*>" word in the template and its correspondig word in the log.

Only Analyse "<*>" word dont analyze other words e.g. [ID],[USER]

Must always key count have to be equal to Template's * count

Always return the output in the format: {{KEY, KEY, ...}}

Now process the following:
Log: {log}
Template: {template}

Output:
"""
from google import genai
import os
from dotenv import load_dotenv

# Path to your .env file
load_dotenv(ENV_FILE) # API KEY .env file location 
client = genai.Client()

results = {}

for idx, row in (df_unique_examples.iterrows()) :
  template = row["template"]
  log = row["log"]
  i = 0
  while i < 5 :
    try :
      response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt.format(log = log , template=template),
      )
      break
    except :
      print(f"failed trying again : {i}. attempt" )
      results.update({template : "failed"})
      i+=1
  print(response.text)
  print("---------------------------------------------------------------------------------------------------------------------------------------------")
  print("--------------------{}---------------".format(template))
  results.update({template : response.text.split("{")[1].split("}")[0]})

results_df = pd.DataFrame(results.items(), columns=["template", "result"])
results_df

from tqdm import tqdm
for idx, row in tqdm(df.iterrows()):
    template = row["template"]
    log_line = row["log"]
    params = template_miner.extract_parameters(template, log_line, exact_matching=True)  # ->
    param_list = results_df[results_df['template']== template]['result'].values[0].split(",") # -> ['DAY', 'TIME', '*', '*', 'USER', '*', 'IP', 'IP', 'IP', 'DURATION', '*', '*']

    param_list = [param.strip() for param in param_list]

    for p_idx , param_type in enumerate(param_list) :
      if param_type != "*" :
        df.at[idx, param_type] = params[p_idx].value

from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

# Map usernames to integer IDs
user_encoder = LabelEncoder()
df["user_id"] = user_encoder.fit_transform(df["USER"].fillna("UNK"))

"""# Template Embedding (template2vec)"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


# ========== Step 2: Generate BERT embeddings ==========
class BertEmbedder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def get_template_vector(self, template_text):
        tokens = self.tokenizer(template_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return cls_embedding

# ========== Example Usage ==========
templates = df["template"].unique()

# Step 1: Build word table
embedder = BertEmbedder()
template_vectors = {}
for t in templates:
    template_vectors[t] = embedder.get_template_vector(t)

"""# Date Encoder param2vec

## Day and Time to Date 
"""

import numpy as np
for idx,row in df.iterrows() :
  date = row["DATE"]
  df.at[idx,"date"] = datetime.strptime(date, "%y%m%d %H%M%S") # YOU CAN CHANGE FOR YOUR OWN TIME TEMPLATE

"""## TIME encode"""

import numpy as np
from datetime import datetime

def time_encoding(value, max_value):
    """Cyclic encoding for a single time unit."""
    angle = 2 * np.pi * value / max_value
    return np.array([np.sin(angle), np.cos(angle)])

def encode_time(time : str) :
    """
    Encode a time string into a cyclic time vector.
    Units: hour, minute, second
    time : %H:%M:%S
    """
    if ":" in time :
      hour , minute , second = map(int , time.split(":"))
    else :
      second = int(time)
      hour = 0
      minute = 0
    encoded = []
    encoded.append(time_encoding(hour , 24))
    encoded.append(time_encoding(minute , 60))
    encoded.append(time_encoding(second , 60))
    return np.concatenate(encoded)

def encode_datetime(dt: datetime):
    """
    Encode a datetime object into a cyclic time vector.
    Units: month, day, hour, minute, second, microsecond (millisecond-level).
    """
    if pd.isna(dt):
        return np.full(12, np.nan) # Return NaN array for missing values
    encoded = []
    # Month (1–12)
    encoded.append(time_encoding(dt.month, 12))
    # Day of month (1–31, rough upper bound)
    encoded.append(time_encoding(dt.day, 31))
    # Hour (0–23)
    encoded.append(time_encoding(dt.hour, 24))
    # Minute (0–59)
    encoded.append(time_encoding(dt.minute, 60))
    # Second (0–59)
    encoded.append(time_encoding(dt.second, 60))
    # Millisecond (0–999 from microsecond field)
    ms = dt.microsecond // 1000
    encoded.append(time_encoding(ms, 1000))

    # Concatenate into one feature vector
    return np.concatenate(encoded)


template_embeds = []
date_embeds = []
connection_time_embeds = []
for idx, row in df.iterrows():
    template = row["template"]
    template_embeds.append(template_vectors[template])

    date_embeds.append(encode_datetime(row["date"]))
    if pd.isna(row["DURATION"]) == False:
      con_time = row["DURATION"]
      if type(con_time) is str   :
        if ":" in con_time :
          conntection_time_sec = (int(con_time.split(":")[0]) * 3600) + (int(con_time.split(":")[1]) * 60) + (int(con_time.split(":")[2]))
          df.at[idx,"DURATION"] = conntection_time_sec
      else :
        df.at[idx,"DURATION"] = int(con_time)

df["template_embed"] = template_embeds
df["date_embed"] = date_embeds

"""## Connection_Time Embedding"""

from sklearn.preprocessing import StandardScaler
duration = df["DURATION"].fillna(0).astype(float).values.reshape(-1, 1)
duration_scaler = StandardScaler()
duration_scaler.fit(duration)
df["duration_scaled"] = duration_scaler.transform(duration)
df["combined_embed"] = df.apply(
    lambda row: np.concatenate([row["template_embed"], row["date_embed"], np.array([row["duration_scaled"]]).reshape(1,)]),
    axis=1
)

df["combined_embed"]

"""# BILSTM With Sliding Window

## Create Sliding Window
"""

df = df.sample(frac=1).reset_index(drop=True)

window_size = 20  # template window, or adjust
sequences = []
n = len(df)
train_end = int(n * 0.8)

df_train = df.iloc[:train_end]
df_test  = df.iloc[train_end:]

df_train.sort_values(by="date", inplace=True)
df_test.sort_values(by="date", inplace=True)

def make_windows(data, window_size=10):
    X, Y, U = [], [], []
    for i in range(len(data) - window_size):
        # input: window of embeddings
        X.append(np.stack(data["combined_embed"].iloc[i:i+window_size]))
        # target: next template embedding
        Y.append(data["combined_embed"].iloc[i+window_size])
        # user: ID of the target log
        U.append(data["user_id"].iloc[i+window_size])
    return np.array(X), np.array(Y), np.array(U)


X_train, Y_train ,U_train   = make_windows(df_train)
X_test,  Y_test , U_test    = make_windows(df_test)

import torch
mu = X_train.mean(axis=(0,1), keepdims=True)   # shape: (1,1,embed_dim)
sigma = X_train.std(axis=(0,1), keepdims=True) + 1e-8

X_train_norm = (X_train - mu) / sigma
X_test_norm  = (X_test  - mu) / sigma

template_dim = df_train["template_embed"].iloc[0].shape[0]  # 768

# compute mu and sigma for template embeddings only
mu_y = Y_train.mean(axis=0, keepdims=True)      # shape (1, template_dim)
sigma_y = Y_train.std(axis=0, keepdims=True) + 1e-8

Y_train_norm = (Y_train - mu_y) / sigma_y
Y_test_norm  = (Y_test  - mu_y) / sigma_y

X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_norm, dtype=torch.float32)

U_train_t = torch.tensor(U_train, dtype=torch.long)
U_test_t  = torch.tensor(U_test, dtype=torch.long)

X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test_norm, dtype=torch.float32)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim*2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden*2)
        energy = torch.tanh(self.attn(lstm_out))   # (batch, seq_len, hidden)
        scores = self.v(energy).squeeze(-1)        # (batch, seq_len)
        attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)

        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        return context, attn_weights

class LogBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1,
                 output_dim=None):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.output_dim = output_dim
        if output_dim:
            self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        if self.output_dim:
            out = self.fc(context)
            return out, attn_weights
        return context, attn_weights

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(torch.tensor(X_train_t, dtype=torch.float32),
                              torch.tensor(Y_train_t, dtype=torch.float32),
                               torch.tensor(U_train_t, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)

model = LogBiLSTM(input_dim=X_train_t.shape[2], hidden_dim=256, output_dim=781 )

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 150
attn_records = [] 

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (X_batch, Y_batch, user_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output, attn_weights = model(X_batch)  # attention dönüyor

        loss = criterion(output, Y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if i == 10:
          print("recorded")
          attn_records.append(attn_weights[0].detach().cpu().numpy())

    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    test_output, attn_weights = model(X_test_t )
    errors = ((test_output - Y_test_t) ** 2).mean(dim=1)  # shape: (num_windows,)

import numpy as np

def explain_anomaly_v2(idx, test_output, Y_test_t ):
    """
    Bir anomaliyi, özellik bazında tahmin hatasını (MSE) hesaplayarak açıklar.

    Args:
        idx (int): Test setindeki anomalinin indeksi.
        test_output (torch.Tensor): Modelin test seti için yaptığı tüm tahminler.
        Y_test_t (torch.Tensor): Test setinin gerçek değerleri.

    Returns:
        tuple: (Hata sözlüğü, Açıklama metni)
    """
    diff = (test_output[idx] - Y_train_t[idx]).cpu().numpy()
    sizes = {"template": 768, "date": 12, "duration": 1} # ADJUST IT 
    start = 0
    errors_mse = {}
    for feature_name, size in sizes.items():
        part = diff[start:start + size]

        errors_mse[feature_name] = np.mean(np.square(part))

        start += size
    main_feature = max(errors_mse, key=errors_mse.get)

    explanation = (
        f"Bu log penceresi anormal çünkü '{main_feature}' özelliğinin tahmini "
        f"beklenenden çok farklı.\n"
        f"Özellik bazında ortalama hata (MSE): {errors_mse}"
    )

    return errors_mse, explanation

threshold = errors.mean() + 2 * errors.std()
anomalies_idx = torch.where(errors > threshold)[0] 
for i in range(len(anomalies_idx)):
  idx = anomalies_idx[i]
  errories, reason = explain_anomaly_v2(idx, test_output, Y_test_t )
  print(errories)  
  print("\n")
  print(reason)  
  print("\n")
  print(df_test.iloc[int(idx)]['log'])
  print("---------------------------------------------------------------------------")

import matplotlib.pyplot as plt

plt.plot(errors.numpy())
plt.xlabel("Window index")
plt.ylabel("MSE error")
plt.title("Prediction error per window")
plt.show()

threshold = errors.mean() + 2*errors.std()
anomalies = torch.where(errors > threshold)[0]
print("Anomalous windows:", anomalies)

anomalous_logs = df_train.iloc[anomalies_idx]

