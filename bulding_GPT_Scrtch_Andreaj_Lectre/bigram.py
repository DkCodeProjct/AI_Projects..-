import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


with open('input.txt', 'r', encoding='utf-8') as file:
    txt = file.read()

#----------------------------
blockSiz = 8
batchSiz = 32
maxSteps = 3000
evalInterval = 300
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evalIters =200
#-----------------------

torch.manual_seed(1337)

# uniq char in list / chars/vocab=65
chars = sorted(list(set(txt)))
vocabSize = len(chars)

# mapping char to int 
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
enc = lambda s: [stoi[c] for c in s]
decod = lambda l: ''.join([itos[i] for i in l])

# Train And Dev/Test Split
data = torch.tensor(enc(txt), dtype=torch.long)
n = int(0.9*(len(data)))
trainData = data[:n]
devData = data[n:]


# get bath/data Loading. getting input||targt batch
def getBatch(split):
    data = trainData if split == 'train' else devData
    ix = torch.randint(len(data) - blockSiz, (batchSiz, ))
    x = torch.stack([data[i:i+blockSiz] for i in ix])
    y = torch.stack([data[i+1:i+blockSiz+1] for i in ix])

    x, y = x.to(device), y.to(device) # give to gpu/cpu

    return x, y


# using noGrad decarator, so itll not BackPorp, and so efficent in cal loss
@torch.no_grad()
def estimatLoss():
    out = {}

    model.eval() # SET to Evaluate mode

    for split in ['train', 'dev']:
        losses = torch.zeros(evalIters)

        for k in range(evalIters):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = loss.mean()

    model.train() # SET Back to Train mode

    return out

# model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocabSiz):
        super(BigramLanguageModel, self).__init__()
        self.tokenEmbTable = nn.Embedding(vocabSiz, vocabSiz)

    def forward(self, ix, targt=None):
        logits = self.tokenEmbTable(ix)

        if targt is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targt = targt.view(B*T)
            loss = F.cross_entropy(logits, targt)

        return logits, loss

    def genarate(self, ix, maxNewTokn):
        for i in range(maxNewTokn):
            logits, loss = self(ix)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            ixNxt = torch.multinomial(probs, num_samples=1)
            ix = torch.cat((ix, ixNxt), dim=1)

        return ix  
    
model = BigramLanguageModel(vocabSize)
m = model.to(device)


optim = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(maxSteps):

    if i % evalIters == 0:
        losses = estimatLoss()
        print(f"step{i}: trainLoss: {losses['train']:.4f}, devLoss: {losses['dev']:.4f}")
    
    xb, yb = getBatch('train')

    logits, loss = model(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

contxt = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decod(m.genarate(contxt, maxNewTokn=500)[0].tolist()))