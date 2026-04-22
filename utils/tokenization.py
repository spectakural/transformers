from tiktoken.core import Encoding
import torch

def text_to_token_ids(text, tokenizer: Encoding):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Adding batch dim with unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer: Encoding):
    flat = token_ids.squeeze(0) # Removing the batch dim with squeeze(0)
    decoded = tokenizer.decode(flat.tolist())
    return decoded