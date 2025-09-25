# Log Anomaly Detection with Template Extraction and BiLSTM-Attention

## Overview
This project implements a log anomaly detection pipeline that combines **Drain3 log template mining**, **LLM-based parameter type extraction**, **BERT-based template embeddings**, and a **BiLSTM with attention mechanism**.  
The pipeline transforms raw log lines into structured embeddings that encode templates, timestamps, and connection durations. These embeddings are then fed into a BiLSTM-attention model for sequence learning and anomaly detection.
---
## Key Features
- **Log Template Mining**: Uses [Drain3](https://github.com/IBM/drain3) to automatically parse logs into templates.
- **Parameter Extraction with LLM**: Employs an LLM (Google Gemini API) to infer parameter types (`USER`, `DATETIME`, `IP`, `DURATION`, `BLKNO`, or `*`) from log templates.
- **Template Embedding**: Encodes templates into dense vectors with a BERT model (`bert-base-uncased`).
- **Time Encoding**: Converts timestamps into cyclic features using sine/cosine transformations for hours, minutes, seconds, days, and months.
- **Connection Time Embedding**: Standardizes connection duration into a scaled numerical feature.
- **BiLSTM with Attention**: Learns sequential dependencies in log data and uses attention weights to highlight important events.
---

## Workflow
1. **Parse Logs with Drain3**  
   - Input: raw log file (e.g., `HDFS.log`)  
   - Output: log templates with placeholders (`<*>`)  

2. **Parameter Type Extraction (LLM)**  
   - For each template, send a prompt to the LLM with the raw log and template.  
   - The LLM outputs a structured mapping of parameter types.  
   - Example:  
     ```
     Log: 081111 092742 25457 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-5952182400333241163 terminating
     Template: <*> <*> <*> dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_<*> terminating
     Output: {{DATETIME, *, *, BLKNO}}
     ```
3. **Embedding Generation**  
   - **Template embeddings**: BERT CLS vectors  
   - **Datetime embeddings**: cyclic encodings of timestamps  
   - **Duration embeddings**: scaled numerical values  

   These are concatenated into a combined feature vector for each log.

4. **Sequence Modeling**  
   - Sliding window approach creates sequences of log embeddings.  
   - BiLSTM with attention predicts the next embedding and highlights important sequence positions.  

---

## Requirements

### Python Dependencies
- Python 3.9+  
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
requirements.txt should contain:
```nginx
torch
transformers
pandas
numpy
tqdm
scikit-learn
matplotlib
drain3
python-dotenv
```
## External Requirements
- **Google Gemini API Key**
  - Required for parameter type extraction.
  - Store the API key in a .env file:
    ```ini
    GEMINI_API_KEY=your_api_key_here
    ```
  - Update **ENV_FILE** in the script to the correct path of `.env`.
---
# How to Run
1. Prepare Logs
  Place your log file (e.g., HDFS.log) in the project directory.
2. Run the Script
    ```
    python DrainAndHDFSwithBLISTM.py
    ```
3. Training 
  The model will train for the specified number of epochs (`num_epochs=500`).
  Intermediate checkpoints are saved as:
    ```
    log_bilstmagain{epoch}_{loss}.pth
    ```
4. Attention Visualization
  Attention weights are recorded during training and can be visualized using `matplotlib`.
# Notes
The LLM parameter extraction step may fail occasionally; retries are implemented automatically.
Ensure logs contain consistent timestamp formats; adjust parsing in datetime.strptime if needed.
The embedding dimension for concatenated vectors must match the model’s input_dim.

<img width="366" height="297" alt="image" src="https://github.com/user-attachments/assets/7bff1d63-fb95-45f9-b19e-a6495f7e6a32" />

Inspired by [**TPLogAD: Unsupervised Log Anomaly Detection Based on Event Templates and Key Parameters**](https://arxiv.org/pdf/2411.15250)

HDFS dataset : [**Logpai Loghub HDFS logs**](https://github.com/logpai/loghub?tab=readme-ov-file)
