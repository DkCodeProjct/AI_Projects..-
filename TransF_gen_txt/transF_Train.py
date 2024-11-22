import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------
# Hyper Para
blockSiz = 256
batchSiz = 64
epochs = 5000
evalInterval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evalIters = 200
nEmb = 384
nHead = 6
nLayers = 5
dropout = 0.2
# ----------

torch.manual_seed(1337)

with open('pg66048.txt', 'r', encoding='utf-8') as file:
    txt = file.read()


chars = sorted(list(set(txt)))
vocabSiz = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
enc = lambda s: [stoi[c] for c in s ]
decod = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(enc(txt), dtype=torch.long)
n = int(0.9*len(data))

trainData = data[:n]
devData = data[n:]

def getBatch(split):
    data = trainData if split == 'train' else devData
    ix = torch.randint(len(data) - blockSiz, (batchSiz, ))
    x = torch.stack([data[i:i+blockSiz] for i in ix])
    y = torch.stack([data[i+1:i+blockSiz+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y 

@torch.no_grad()
def estimateLoss():
    out = {}
    model.eval()
    for split in ['train', 'dev']:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

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
        
        k = self.key(x)
        q = self.quary(x)

        w = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        v = self.value(x)
        out = w @ v

        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, nHead, headSiz):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(headSiz) for i in range(nHead)])
        self.projection = nn.Linear(headSiz * nHead, nEmb)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        
        return out  

class FeedForwardNetowrk(nn.Module):
    def __init__(self, nEmb):
        super(FeedForwardNetowrk, self).__init__()
        self.net = nn.Sequential(
            # 4*nEmb:
            #  ,In atten Paper, inner lyr of the FFN, Should mul by 4 in terms of channle siz
            nn.Linear(nEmb, 4 * nEmb),
            nn.ReLU(),
            nn.Linear(4 * nEmb, nEmb),
            nn.Dropout(dropout),
        )

    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, nEmb, nHead):
        super(Block, self).__init__()
        headSiz = nEmb // nHead
        self.selfAttn = MultiHeadAttention(nHead, headSiz)
        self.ffn = FeedForwardNetowrk(nEmb)
        self.layerNorm_1 = nn.LayerNorm(nEmb)
        self.layerNorm_2 = nn.LayerNorm(nEmb)
    
    def forward(self, x):
        x = x + self.selfAttn(self.layerNorm_1(x))
        x = x + self.ffn(self.layerNorm_2(x))

        return x 

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.toknEmbTable = nn.Embedding(vocabSiz, nEmb)
        self.posEmbTable = nn.Embedding(blockSiz, nEmb)
        self.blocks = nn.Sequential(*[Block(nEmb, nHead=nHead) for i in range(nLayers)])
        self.finlLayrNorm = nn.LayerNorm(nEmb)
        self.lmHead = nn.Linear(nEmb, vocabSiz)

        self.apply(self._init__weights)
    
    def _init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   
    def forward(self, ix, targt=None):
        B, T = ix.shape

        tokEmb = self.toknEmbTable(ix)
        posEmb = self.posEmbTable(torch.arange(T, device=device))
        x = tokEmb + posEmb
        x = self.blocks(x)
        x = self.finlLayrNorm(x)
        logits = self.lmHead(x)

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
            ixCond = ix[:, -blockSiz:]
            logits, loss = self(ixCond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            ixNxt = torch.multinomial(probs, num_samples=1)

            ix = torch.cat((ix, ixNxt), dim=1)

        return ix

model = BigramLanguageModel()
m = model.to(device=device)

print(sum(p.numel() for p in m.parameters()))

optim = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(epochs):

    if i % evalInterval == 0 or i == epochs-1:
        losses = estimateLoss()
        print(f"step {i}, trainLoss: [ {losses['train']:.4f} ], devLoss[ {losses['dev']:.4f} ]")
    
    xb, yb = getBatch('train')

    logits, loss = model(xb, yb)
    optim.zero_grad()
    loss.backward()
    optim.step()


contxt = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decod(m.genarate(contxt, maxNewTokn=500)[0].tolist()))

