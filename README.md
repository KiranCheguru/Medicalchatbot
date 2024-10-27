# Medicalchatbot
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("D:/Internship/Chat bot Project/intents.json"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#####

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
print('import done')

# Load the trained tokenizer and model

path = "D:/Internship/Chat bot Project/medical-chat-bot-pytorch-120epochs-v1"
tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPT2LMHeadModel.from_pretrained(path)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Number of parameters in the model: {num_params/10**6}M")

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Provide context (including your question)
context = """
I have a fever and a runny nose.
"""

# Tokenize the context
input_ids = tokenizer.encode(context, return_tensors='pt').to(device)

# Generate text based on the context
output = model.generate(input_ids, max_length=75, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode the output tokens back into text
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("120 Epoch Model's Response:")
print(response)
#####

# Load the trained tokenizer and model
path = "D:/Internship/Chat bot Project/medical-chat-bot-pytorch-500epochs-v1"
tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPT2LMHeadModel.from_pretrained(path)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Number of parameters in the model: {num_params/10**6}M")

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Provide context (including your question)
context = """
I have a fever and a runny nose.
"""

# Tokenize the context
input_ids = tokenizer.encode(context, return_tensors='pt').to(device)

# Generate text based on the context
output = model.generate(input_ids, max_length=75, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode the output tokens back into text
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("500 epoch Model's Response:")
print(response)

