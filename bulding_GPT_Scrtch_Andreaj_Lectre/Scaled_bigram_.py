import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


with open('input.txt', 'r', encoding='utf-8') as file:
    txt = file.read()

#----------------------------
blockSiz = 256
batchSiz = 64

maxSteps = 5000
evalInterval = 500
nEmb = 384
nHead = 6
nLayers = 6
lr = 3e-4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evalIters =200
#-----------------------

torch.manual_seed(1337)

# uniq char in list / chars/vocab=65
chars = sorted(list(set(txt)))
vocabSiz = len(chars)

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



# head Module
class Head(nn.Module):
    def __init__(self, headSiz):
        super(Head, self).__init__()
        self.key = nn.Linear(nEmb, headSiz, bias=False)
        self.quary = nn.Linear(nEmb, headSiz, bias=False)
        self.value = nn.Linear(nEmb, headSiz, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blockSiz, blockSiz)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C) 
        q = self.quary(x) # (B, T, C)

        #compute the atten scores ('Affinities')

        w = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        w = F.softmax(w, dim=-1) # (B, T, T)
        w = self.dropout(w)
        #perfom the Weighted Aggreation of the val
        v = self.value(x) # (B, T, C)

        out = w @ v # (B, T, T) @ (B, T, C) --> (B, T, C)

        return out 

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, numHead, headSiz):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(headSiz) for i in range(numHead)]) 
        self.projection = nn.Linear(nEmb, nEmb)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        # Projction is just a Linear Transformation of the out 
        out = self.projection(out)
        return out 

# FEED FORWARD NETWORD . FFN
class FeedForwardNetword(nn.Module):
    def __init__(self, nEmb):
        super(FeedForwardNetword, self).__init__()
        
        self.net = nn.Sequential(
            # 4*nEmb:
            #     In atten Paper, inner lyr of the FFN, Should mul by 4 in terms of channle siz
            nn.Linear(nEmb, 4 * nEmb),
            nn.ReLU(),
            nn.Linear(4 * nEmb, nEmb), # projction lyr going back to the Risidual Pathway
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# Transofrmr Block: Communication Followd by Computation 
class Block(nn.Module):
    def __init__(self, nEmb, nHead):
        super(Block, self).__init__()
        
        # nEmb: emb dim, nHead: num of heads we'd like 
        headSiz = nEmb // nHead
        self.selfAttn = MultiHeadAttention(nHead, headSiz)
        self.ffn = FeedForwardNetword(nEmb)
        self.layrNorm_1 = nn.LayerNorm(nEmb)
        self.layrNorm_2 = nn.LayerNorm(nEmb)
    def forward(self, x):
        
        # Risidual Connection x = x + .... 
        x = x + self.selfAttn(self.layrNorm_1(x))
        x = x + self.ffn(self.layrNorm_2(x))
        return x 


# model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.tokenEmbTable = nn.Embedding(vocabSiz, nEmb)
        self.posEmbTable = nn.Embedding(blockSiz, nEmb)
        
        self.blocks = nn.Sequential(*[Block(nEmb, nHead=nHead) for _ in range(nLayers)])
        self.layrNorm = nn.LayerNorm(nEmb)

        self.lmHead = nn.Linear(nEmb, vocabSiz)


    def forward(self, ix, targt=None):
        B, T = ix.shape

        tokenEmb = self.tokenEmbTable(ix) # (B, T, C)
        posEmb = self.posEmbTable(torch.arange(T, device=device)) # (T, C)
        x = tokenEmb + posEmb
        x = self.blocks(x)
        x = self.layrNorm(x)
        logits = self.lmHead(x) # (B, T, vocabSiz)
        
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
            # crop  indx to the last blockSiz token
            ixCond = ix[:, -blockSiz:]

            logits, loss = self(ixCond)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            ixNxt = torch.multinomial(probs, num_samples=1)
            ix = torch.cat((ix, ixNxt), dim=1)

        return ix  
    
model = BigramLanguageModel()
m = model.to(device)

#print/.count para
print(sum(p.numel() for p in m.parameters())/1e6, 'para')

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

#########################################
# Loss LOG
#
# _- With One self Attn [Head]
#        : trainLoss: 2.3243, devLoss: 2.3136
#
# _- With Multi Head Self Attn:
#        : trainLoss: 2.0762, devLoss: 2.1372
#
# _- With ffn/ Block/ Risidual Connection:
#        : trainLoss: 2.0787, devLoss: 2.2008
#
#  _- With HyperPara/Block/FFN ::::
#     NOTE:  
#     * The loss is high cos i fogot to add:
#         + block and layerNorm in forward pass ./
#                x = self.blocks(x)
#                x = self.layrNorm(x)
#    ++ Kind of crazy with Those 2 layers itll get 15m or so to train, without it its like 8m
#    : trainLoss: 2.4613, devLoss: 2.4811
#     // however i didnt train andrej metion it,, its take so long on GoogleColab so ill do it later :/ ¯\_(ツ)_/¯ 
#########################################

### NOTE ###:
# 
# Also we can see our Train Loss is Getting Ahead of out Dev Loss 
#    :Its cos,
#       + our Model Getting big Enough, and Little bit of Overfitting....
#