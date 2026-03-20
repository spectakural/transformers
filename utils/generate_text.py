import torch
import torch.nn as nn

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        cropped_idx = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(cropped_idx)
        logits = logits[:,-1,:] # Only taking the last token logits as they contain the next word
        prob = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(prob, dim=-1, keepdim=True)
        # print(f"[DEBUG] idx shape: {idx.shape} | idx value: {idx}")
        # print(f"[DEBUG] idx_next shape: {idx_next.shape} | idx_next value: {idx_next}")
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx