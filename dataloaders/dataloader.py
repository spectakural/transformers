import tiktoken
import torch 
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_tokens = []
        self.target_tokens = []
        
        token_ids = tokenizer.encode(txt, allowed_special=["<|end_of_text|>"])
        
        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i: i + max_len]
            target_chunk = token_ids[i + 1: i + max_len + 1]
            self.input_tokens.append(torch.tensor(input_chunk))
            self.target_tokens.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_tokens)
    
    def __getitem__(self, index):
        return self.input_tokens[index], self.target_tokens[index]
    
    
def create_dataloader_v1(
    txt, max_len, stride,
    batch_size, shuffle=True, num_workers=0, drop_last=True
):
    # Initialize the gpt2 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create the dataset with GPTDatasetV1 Class
    dataset = GPTDatasetV1(txt, tokenizer, max_len, stride)
    
    # Create a dataloader from pytorch
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )
    
    return dataloader