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


# Model Params Calculation
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / 1024 / 1024
print(f"Total size of the model: {total_size_mb:.2f} MB")


# Sample inference with generate_text_simple fn
from utils.generate_text import generate_text_simple

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print(f"\n\nEncoded Start Context: {encoded}")
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(f"encoded_tensor shape: {encoded_tensor.shape}")

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_124M_CONFIG.context_length
)
print(f"\nOutput: {out}")
print(f"Output length: {len(out[0])}")

# decoding the text
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(f"Decoded response: {decoded_text}")