import tiktoken
import torch
from models.GPTModel import GPTModel
from setting import GPT_124M_CONFIG

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = GPTModel(GPT_124M_CONFIG)

out = model(batch)
print("Input batch: \n", batch)
print("\nOutput shape: ", out.shape)
print(out)