import os
import torch
import tiktoken
from dataloaders.dataloader import create_dataloader_v1
from setting import GPT_124M_CONFIG


file_path = "dataset/the-verdict.txt"
tokenizer = tiktoken.get_encoding("gpt2")

with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
    
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)
print("Tokens:", total_tokens)
    
    
train_ratio = 0.9
split_idx = int(len(text_data) * train_ratio)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    txt = train_data,
    batch_size=2,
    max_len=GPT_124M_CONFIG.context_length,
    stride = GPT_124M_CONFIG.context_length,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    txt = val_data,
    batch_size=2,
    max_len=GPT_124M_CONFIG.context_length,
    stride = GPT_124M_CONFIG.context_length,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

# Sanity check

if total_tokens * (train_ratio) < GPT_124M_CONFIG["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_124M_CONFIG['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_124M_CONFIG["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_124M_CONFIG['context_length']` or "
          "decrease the `training_ratio`")
    
    
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)


train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)
    
