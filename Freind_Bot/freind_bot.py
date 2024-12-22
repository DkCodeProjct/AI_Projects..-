import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
#from transformers import GPT2Tokenizer

"""
import pandas as pd
# Load the file
input_file = '/mnt/data/dialogs.txt'
output_file = '/mnt/data/labeled_dialogs.csv'
df = pd.read_csv(input_file, sep='\t', names=["Context", "Response"], skiprows=1)
df["Context"] = "Context: " + df["Context"]
df["Response"] = "Response: " + df["Response"]
df.to_csv(output_file, index=False)
print(f"Labeled dataset saved to {output_file}")
"""

batchsiz = 64
blocksiz = 128
epochs = 600
evalIntervals = 100
lr = 3e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
evaliters = 50
nemb = 112
nhead = 2
nlayers = 1
dropout = 0.3

"""
batchsiz = 64
blocksiz = 128
epochs = 600
evalIntervals = 100
lr = 3e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
evaliters = 100
nemb = 112
nhead = 2
nlayers = 2
dropout = 0.1
"""

with open("dialogs.txt", 'r', encoding="utf-8") as file:
    txt = file.read()

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

def enc(txt, tok):
    tokns = tok(txt, return_tensors="pt", truncation=True, padding=False)["input_ids"]
    return tokns.flatten()

data = torch.tensor(enc(txt, tokenizer), dtype=torch.long)

n = int(0.9*len(data))
trainData = data[:n]
valData = data[n:]

vocabsiz = tokenizer.vocab_size
print(f"vocab siz: {vocabsiz}")

def getBatch(split):
    dataset = trainData if split == 'train' else valData
    ix = torch.randint(0, len(dataset) - blocksiz, (batchsiz,))

    x = torch.stack([dataset[i:i+blocksiz] for i in ix])
    y = torch.stack([dataset[i+1:i+blocksiz+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y 

def estimateLoss():
    out = { }
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(evaliters)
        for k in range(evaliters):
            x, y = getBatch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out 

class Head(nn.Module):
    def __init__(self, headsiz):
        super().__init__()
        self.key = nn.Linear(nemb, headsiz, bias=False)
        self.quary = nn.Linear(nemb, headsiz, bias=False)
        self.value = nn.Linear(nemb, headsiz, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(blocksiz, blocksiz)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.quary(x)

        w = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        v = self.value(x)
        out = w @ v
         
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, nhead, headsiz):
        super().__init__()
        self.heads = nn.ModuleList([Head(headsiz) for _ in range(nhead)])
        self.proj = nn.Linear(headsiz * nhead, nemb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads])
        out = self.dropout(self.proj(x))
        return out 

class FeedForwadNetwork(nn.Module):
    def __init__(self, nemb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nemb, 4 * nemb), 
            nn.ReLU(),
            nn.Linear(4 * nemb, nemb),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, nemb, nhead):
        super().__init__()
        headsiz = nemb // nhead
        self.self_attn = MultiHeadAttention(nhead, headsiz)
        self.ffn = FeedForwadNetwork(nemb)
        self.ln_1 = nn.LayerNorm(nemb)
        self.ln_2 = nn.LayerNorm(nemb)
    
    def forward(self, x):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x 

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocabsiz, nemb)
        self.wpe = nn.Embedding(blocksiz, nemb)
        self.block = nn.Sequential(*[Block(nemb, nhead=nhead) for _ in range(nlayers)])
        self.ln_finl = nn.LayerNorm(nemb)
        self.lm_Head = nn.Linear(nemb, vocabsiz)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, ix, targt=None):
        B, T = ix.shape
        tokEmb = self.wte(ix)
        posEmb = self.wpe(torch.arange(T, device=device))

        x = tokEmb + posEmb
        for block in self.block:
            x = block(x)
        x =  self.ln_finl(x)

        logits = self.lm_Head(x)

        if targt is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targt = targt.view(B*T)
            loss = F.cross_entropy(logits, targt, label_smoothing=0.1)

        return logits, loss 
    
    def genarate(self,ix, maxNewTok, tokenizer):
        # idx is (B, T) array of indices in the current context
        for _ in range(maxNewTok):
            # crop idx to the last block_size tokens
            ixCond = ix[:, -blocksiz:]

            # predict
            logits, loss = self(ixCond)
            
            # focus only on the last time step
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            ixNxt = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            ix = torch.cat((ix, ixNxt), dim=1)

        genTxt = tokenizer.decode(ix[0].cpu().numpy().tolist(), skip_special_tokens=True)
        return genTxt

model = GPTLanguageModel()
m = model.to(device)
# Use Torch.Compinle,, well Expect that fucking Error
useCompile = False
if useCompile:
    model = torch.compile(model)
    print("using Torch Compile")

optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) #add weight decay to avoid overfit
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, verbose=True) 

trainLoss = []
valLoss = []
xVal = []
for i in range(epochs):
    if i % evalIntervals == 0 or i == epochs - 1:
        losses = estimateLoss()
        trainLoss.append(losses["train"].item())
        valLoss.append(losses["val"].item())
        xVal.append(i)
        print(f"Step {i} | train loss {losses['train']:.4f} | val loass {losses['val']:.4f}")
    
    xb, yb = getBatch("train")
    logits, loss = model(xb, yb)

    optim.zero_grad()
    loss.backward()
    optim.step()

def saveCheckpnt(model, optimizer, epoch, loss, filepath):
    checkPnt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkPnt, filepath)
    print(f"Checkpoint saved to {filepath}")

# Saving model checkpoint
saveCheckpnt(model, optim, epochs-1, valLoss[-1], "FreindBotModelTrainFinl.pth")


#generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Initial context
genTxt = model.genarate(context, maxNewTok=500, tokenizer=tokenizer)
print(genTxt) 


#####################
#################
############# Fine Tune THe ModeL
#################
#####################


import matplotlib.pyplot as plt

#plt.plot(lossi)

import matplotlib.pyplot as plt

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot( xVal, trainLoss, label="Train loss")
plt.plot( xVal, valLoss,  label="Val loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train/Val Loss")
plt.legend()
plt.grid(False)
plt.show()