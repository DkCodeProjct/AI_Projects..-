import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

with open("/content/reddit_text-davinci-002.csv", 'r', encoding="utf-8") as file:
    txt = file.read()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#pad token
tokenizer.pad_token = tokenizer.eos_token

encoded = tokenizer(txt, return_tensors="pt", truncation=True, padding=True)
data = encoded['input_ids'].squeeze()  # Get the input_ids tensor
vocab_siz = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_siz}")

# ////////////////////////////////////////
# ////////////////////////////////////////
# ////////////////////////////////////////

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.nemb, 3 * config.nemb)
        self.c_proj = nn.Linear(config.nemb, config.nemb)

        self.nHead = config.nhead
        self.nemb = config.nemb

        self.register_buffer("bias", torch.tril(torch.ones(config.blocksiz, config.blocksiz)).view(1, 1, config.blocksiz, config.blocksiz))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.nemb, dim=2)
        k = k.view(B, T, self.nHead, C // self.nHead).transpose(1, 2)
        q = q.view(B, T, self.nHead, C // self.nHead).transpose(1, 2)
        v = v.view(B, T, self.nHead, C // self.nHead).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.nemb, 4 * config.nemb)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.nemb, config.nemb)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.nemb)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.nemb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class GPTConfig:
    blocksiz :int = 512
    nemb :int = 256
    nhead :int = 8
    n_layers :int = 8
    batch_siz = 64

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_siz, config.nemb),
            wpe = nn.Embedding(config.blocksiz, config.nemb),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.nemb)
        ))

        self.lm_head = nn.Linear(config.nemb, vocab_siz, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, ix, targt=None):
        B, T = ix.size()
        assert T <= self.config.blocksiz, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=ix.device)
        posEmb = self.transformer.wpe(pos)
        tokEmb = self.transformer.wte(ix)
        x = posEmb + tokEmb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None

        if targt is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targt.view(-1))

        return logits, loss

#/////////////////////////////////////
#/////////////////////////////////////
#/////////////////////////////////////

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch to the same length.
    """
    x, y = zip(*batch)  # Unpack the batch into inputs (x) and targets (y)

    # Pad the sequences to the maximum length in the batch
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)  # Adjust padding_value as needed
    y_padded = pad_sequence(y, batch_first=True, padding_value=-100)  # Commonly used for loss masking

    return x_padded, y_padded
class TextDataset(Dataset):
    def __init__(self, tokenizer, txt, block_size):
        self.tokenizer = tokenizer
        self.tokens = tokenizer(txt, return_tensors="pt", truncation=True, padding=False)['input_ids'].squeeze()
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        x = torch.tensor(self.tokens[start:end-1])  # Input sequence
        y = torch.tensor(self.tokens[start+1:end])  # Target sequence
        return x, y


dataset = TextDataset(tokenizer, txt, GPTConfig.blocksiz)

dataloader = DataLoader(
    dataset,
    batch_size=GPTConfig.batch_siz,  # Adjust the batch size as needed
    shuffle=True,
    collate_fn=collate_fn  # Use the custom collate function
)
model = GPT(GPTConfig)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)


epochs = 1000
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, loss = model(x, targt=y)
        total_loss += loss.item()

        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
    # Save checkpoint   

final_model_path = "final_gpt_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final trained model saved to {final_model_path}")

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the dataset
df = pd.read_csv("/content/reddit_text-davinci-002.csv")  # Path to your dataset
print(df.head())

class FineTuneDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the prompt and completion
        prompt = self.data.iloc[idx]['prompt']
        completion = self.data.iloc[idx]['completion']
        
        # Concatenate prompt and completion
        full_text = f"{prompt} {self.tokenizer.eos_token} {completion}"
        
        # Tokenize and truncate to block size
        tokens = self.tokenizer(full_text, truncation=True, max_length=self.block_size, return_tensors="pt", padding="max_length")
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()

        # Create labels
        labels = input_ids.clone()
        labels[:len(self.tokenizer(prompt)["input_ids"])] = -100  # Ignore prompt tokens for loss

        return input_ids, labels, attention_mask

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

# Hyperparameters
BLOCK_SIZE = 256
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 20

# Create dataset and dataloader
dataset = FineTuneDataset(df, tokenizer, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load your pre-trained GPT model
model = GPT(GPTConfig)
model.load_state_dict(torch.load("final_gpt_model.pth"))  # Load pre-trained weights
model.to(device)

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Fine-tuning loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (input_ids, labels, attention_mask) in enumerate(dataloader):
        input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)

        # Forward pass
        logits, loss = model(input_ids, targt=labels)  # `targt` should match label tensor
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

# Save fine-tuned model
fine_tuned_model_path = "fine_tuned_gpt_model.pth"
torch.save(model.state_dict(), fine_tuned_model_path)
print(f"Fine-tuned model saved to {fine_tuned_model_path}")
