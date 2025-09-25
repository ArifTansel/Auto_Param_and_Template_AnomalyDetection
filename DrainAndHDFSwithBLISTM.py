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
combined = np.vstack([
    np.concatenate([
        row["template_embed"],
        row["date_embed"],
        np.array([row["duration_scaled"]])
    ])
    for _, row in df.iterrows()
])


import numpy as np

# tüm veriyi RAM'e alıyoruz
mu = combined.mean(axis=0, keepdims=True)   # ortalama
sigma = combined.std(axis=0, keepdims=True) + 1e-8  # std + epsilon


import torch
from torch.utils.data import Dataset, DataLoader

class LogDataset(Dataset):
    def __init__(self, array, window_size, mu=None, sigma=None):
        # Burada doğrudan RAM’deki numpy array’i kullanıyoruz
        self.embeds = array
        self.window_size = window_size
        self.mu = mu
        self.sigma = sigma

    def __len__(self):
        return len(self.embeds) - self.window_size

    def __getitem__(self, idx):
        X = (self.embeds[idx:idx+self.window_size] - self.mu) / self.sigma
        Y = (self.embeds[idx+self.window_size] - self.mu) / self.sigma
        return torch.from_numpy(X).float(), torch.from_numpy(Y).float().squeeze()

# Kullanım
window_size = 20
dataset = LogDataset(combined, window_size, mu, sigma)  # memmap yerine RAM'deki array
loader = DataLoader(dataset, batch_size=1024, shuffle=False,pin_memory=True, num_workers=8 ,persistent_workers=True)



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

model = LogBiLSTM(input_dim=781, hidden_dim=256, output_dim=781 )

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 150
attn_records = [] 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # sabit LR

num_epochs = 500
attn_records = []

for epoch in range(num_epochs):
    total_loss = 0
    for i, (X_batch, Y_batch) in tqdm(enumerate(loader), total=len(loader)):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        optimizer.zero_grad()
        output, attn_weights = model(X_batch)  # attention dönüyor
        loss = criterion(output, Y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # sadece 1 kez çağır
        total_loss += loss.item()
        if i == 10:  # attention kaydet
            attn_records.append(attn_weights[0].detach().cpu().numpy())
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(loader):.4f}")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"log_bilstmagain{epoch}_{total_loss/len(loader):.4f}.pth")


