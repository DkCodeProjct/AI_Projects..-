import torch
import torch.nn.functional as F
import random

#dataset
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
vocabSiz = len(itos)


def buildDataset(words):
    blocksiz = 3
    X, Y = [], []
    for w in words:
        contxt = [0] * blocksiz
        for ch in w + '.':
            ix = stoi[ch]
            X.append(contxt)
            Y.append(ix)
            contxt = contxt[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

xTrain, yTrain = buildDataset(words[:n1])
xDev, yDev = buildDataset(words[n1:n2])
xTst, yTst = buildDataset(words[:n2])


nEmb = 10
nHidden = 200
blocksiz = 3

g = torch.Generator().manual_seed(2147483647)
C = torch.randn(vocabSiz, nEmb,              generator=g)
w1 = torch.randn((nEmb * blocksiz), nHidden, generator=g )
b1 = torch.randn(nHidden,                    generator=g)
w2 = torch.randn((nHidden, vocabSiz),        generator=g)
b2 = torch.randn(vocabSiz,                   generator=g)
para = [C, w1, b1, w2, b2]

for p in para:
    p.requires_grad = True


maxStep = 150000 #200000 if have fancy machine
batchSiz = 32
loss_i = []

for i in range(maxStep):
    ix = torch.randint(0, xTrain.shape[0], (batchSiz,), generator=g)
    xb, yb = xTrain[ix], yTrain[ix]

    emb = C[xb]
    embCat = emb.view(emb.shape[0], -1)
    hidnLyerPreActiv = embCat @ w1 + b1
    h = torch.tanh(hidnLyerPreActiv)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, yb)

    for p in para:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < (maxStep%2==0) else 0.01
    for p in para:
        p.grad += -lr * p.grad
    
    if i % 10000 == 0:
        print(f'{i:7d}/{maxStep:7d}: {loss.item():4.f}')
    loss_i.append(loss.log10().item())

@torch.no_grad()
def splitLoss(split):
    x, y = {
        'train':(xTrain, yTrain),
        'dev':(xDev, yDev),
        'test':(xTst, yTst)
    }[split]

    emb = C[x]
    embCat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embCat @ w1 + b1)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

splitLoss('train')
splitLoss('dev')

g = torch.Generator().manual_seed(2147483647 + 10)
blocksiz=3
for _ in range(10):
    out = []
    contxt = [0] * blocksiz
    while True:
        emb = C[torch.tensor([contxt])]
        h = torch.tanh(emb.view[1, -1] @ w1 + b1)
        probs = torch.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        contxt = contxt[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))


