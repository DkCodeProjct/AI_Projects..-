    # Reproduce GPT. Andrej Karpathy Lecture

import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os

from torch.nn.parallel import DistributedDataParallel as DDP
# ===============================


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.nEmb % config.nHead == 0

        # K, Q, V Projections for all heads, but in batch
        self.c_attn = nn.Linear(config.nEmb, 3 * config.nEmb)

        # output projection
        self.c_proj = nn.Linear(config.nEmb, config.nEmb)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # dealing with Residual Growth
        # Regularization
        self.nHead = config.nHead
        self.nEmb = config.nEmb

        # not realy a biase, more of a Mask, but follwing OpenAi/HF Naming Though
        self.register_buffer("bias", torch.tril(torch.ones(config.blockSiz, config.blockSiz)).view(1, 1, config.blockSiz, config.blockSiz))

    def forward(self, x):
        B, T, C = x.shape #batchSiz, seqLen, embDim[(nEmb)]

        #calc K,Q,V for all heads in batch and movie head forwrd to be the batch dim
        # nh==numHead,  hs==headSiz, C==numOfChannels == nh * hs
        #eg: in GPT2(124M) nHead=12, hs=64, so nh*hs==C[768]

        Q_K_V = self.c_attn(x)
        q, k, v = Q_K_V.split(self.nEmb, dim=2)
        k = k.view(B, T, self.nHead, C // self.nHead).transpose(1, 2) # B, nh, T, hs
        q = q.view(B, T, self.nHead, C // self.nHead).transpose(1, 2) # B, nh, T, hs
        v = v.view(B, T, self.nHead, C // self.nHead).transpose(1, 2) # B, nh, T, hs


        # So theres Some optim that torch.compile Potentialy Cannot Find, so We use FlashAttention
        #attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        #attn = F.softmax(attn, dim=-1)
        #y =  attn @ v # (B, nh, T, T) x (B, nh, T, hs) ==> (B, nh, T, hs)

        # flashAttention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.c_proj(y)

        return y



class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.nEmb, 4 * config.nEmb) # fuly conected
        self.gelu = nn.GELU(approximate='tanh') # Read the paper for Aproxiation
        self.c_proj = nn.Linear(4 * config.nEmb, config.nEmb)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # dealing with Residual Growth

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.nEmb)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.nEmb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    blockSiz: int = 1024 #max seq len

    #numOfToken 5057: 50K BPE mergs + 256 byte Token + 1<|endoftext|>
    vocabSiz: int = 50257
    nLayers: int = 12 # num Of layrs
    nHead: int = 12 # num of heads
    nEmb: int = 768 # emb dim



class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocabSiz, config.nEmb),
            wpe = nn.Embedding(config.blockSiz, config.nEmb),
            h = nn.ModuleList([Block(config) for _ in range(config.nLayers)]),
            ln_f = nn.LayerNorm(config.nEmb),
        ))

        self.lm_head = nn.Linear(config.nEmb, config.vocabSiz, bias=False)

        # init para
        self.apply(self._init_weights)

    def _init_weights(self, moduel):
        if isinstance(moduel, nn.Linear):
            std = 0.02
            # We have 2 risidual path going //mlp, attn\\ so 2*
            if hasattr(moduel, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.nLayers) **-0.5
            torch.nn.init.normal_(moduel.weight, mean=0.0, std=std)
            if moduel.bias is not None:
                torch.nn.init.zeros_(moduel.bias)

        elif isinstance(moduel, nn.Embedding):
            torch.nn.init.normal_(moduel.weight, mean=0.0, std=0.02)

        ####
        # Loading para from the hugging face to our code, and init GPT class with those para
        ####


    def forward(self, ix, target=None):
        # index is always of shape B, T, }} batch dim and time dim
        B, T = ix.size()
        assert T <= self.config.blockSiz, f"cannot forward seq of len {T}, blockSiz,"

        pos = torch.arange(0, T, dtype=torch.long, device=ix.device)
        posEmb = self.transformer.wpe(pos)
        tokEmb = self.transformer.wte(ix)
        x = posEmb + tokEmb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if target is not None:

            # Flatting out Our [B,T,C] input Tensor to -> [B, T], COS Cros enrophy didnt line multi Dim tensors
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, modelTyp):
        """ Loading pre-trained GPT-2 model W from huging face """
        assert modelTyp in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading W from pretrained gpt: %s" % modelTyp)

        configArgs = {
            'gpt2':         dict(nLayers=12, nHead=12, nEmb=768),
            'gpt2-medium':  dict(nLayers=24, nHead=16, nEmb=1024),
            'gpt2-large':   dict(nLayers=36, nHead=20, nEmb=1080),
            'gpt2-xl':      dict(nLayers=48, nHead=25, nEmb=1600),
        }[modelTyp]

        configArgs['vocabSiz'] = 50257
        configArgs['blockSiz'] = 1024

        # Create a from-scratch init minGPT model
        config = GPTConfig(**configArgs)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(modelTyp)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]# ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]# same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configOptim(self, weightDecay, lr, device):
        #start with the all of the candidate para (that reqar grad)
        paraDict = {pn:p for pn, p in self.named_parameters()}
        paraDict = {pn:p for pn, p in paraDict.items() if p.requires_grad}

        #create optim groups, Any para that is 2d will be weight decay, else no
        #ex: all weight tensor in matmul + emb decay, all bias and layrNorm dont
        decayPara = [p for n, p in paraDict.items() if p.dim() >= 2]
        noDecayPara = [p for n, p in paraDict.items() if p.dim() < 2]
        optimGroups = [
            {"params":decayPara, "weightDecay":weightDecay},
            {"params":noDecayPara, "weightDecay":0.0}
        ]

        numDecayPara = sum(p.numel() for p in decayPara)
        numNoDecayPara = sum(p.numel() for p in noDecayPara)

        print(f"num decay para tensor: {len(decayPara)}, with {numDecayPara:,}, para")
        print(f"num non decay para tensor: {len(noDecayPara)}, with {numNoDecayPara:,}, para")

        # create AdamW optim and use Fused version if it is availble
        # in torch doc AdamW dose not contain fuesed it add later, so thta why we Inspectiing
        fusedAvailble = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        useFused = fusedAvailble and 'cuda' in device
        print(f"using fused AdamW {useFused}")

        optim = torch.optim.AdamW(optimGroups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=useFused)

        return optim


#----------------------------
import tiktoken

class DataLoader:
    def __init__(self, B, T, processRank, numProces):
        self.B = B
        self.T = T
        self.processRank = processRank
        self.numProces = numProces


        with open('/content/input.txt', 'r') as file:
            txt = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(txt)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B * T)} bathes")

        #state
        self.currentPos = B * T * self.processRank

    def nextBatch(self):
        B, T = self.B, self.T
        # Getting Targt Token for Input Tokenes
        buf = self.tokens[self.currentPos : self.currentPos+B*T+1]
        x = (buf[:-1].view(B, T))# input
        y = (buf[1:].view(B, T))# targt/lable

        self.currentPos += B * T * self.numProces

        if self.currentPos + (B * T * self.numProces + 1) > len(self.tokens):
            self.currentPos = self.B * self.T * self.processRank

        return x, y


import tim

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f'Using {device}')


# run the training loop
from torch.distributed import init_process_group, destroy_process_group

# /////////////////
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
# NOTE: Im just copy pase andrej code, i dont think  id have this many fucking GPUs ever [maybe in future ill effort]


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Greadient Accumilation:
# For simulate [0.5M BatchSiz] like in papaer
totalBatchSiz = 524288 # 2**19 , ~0.5M, in num of tok
B = 16 # micro batch siz
T = 1024 # seq len
assert totalBatchSiz % (B * T * ddp_world_size) == 0, "make sure total batch Size is devisible by B * T * ddp_world_size"
gradAccumStep = totalBatchSiz // (B * T * ddp_world_size)

#if master_process: # else This would Print x times Depending on how many GPUs you;ve
#    print(f'total desired batch siz: {totalBatchSiz}')
#    print(f"=> calc grad accum step: {gradAccumStep}")

trainLoader = DataLoader(B=B, T=T, processRank=ddp_rank, numProces=ddp_world_size) #i ran out of mem
# B=8, T=512
print("im GPU ", ddp_rank)
import sys; sys.exit(0)

# Enabeling  TF-32
torch.set_float32_matmul_precision('high')

# Create model
model = GPT(GPTConfig(vocabSiz=50304)) # GPU like when it get num of pow of 2, even.,, 50304 is a pow of 2
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=(ddp_local_rank))


#Cosine Decay
maxLr = 6e-4
minLr = maxLr * 0.1
warmupSteps = 10
maxSteps = 2 #50

# Learnig Rate Schedule
def getLearningRate(it):
    # 1. Linear warmup for warmup_iters steps
    if it < warmupSteps:
        return maxLr * (it+1) / warmupSteps

    # 2. if it > lr_decay_iters, return min lr
    if it > maxSteps:
        return minLr

    # 3. in between, use cosine decay down to min lr
    decatRatio = (it - warmupSteps) / (maxSteps - warmupSteps)
    assert 0 <= decatRatio <= 1

    coEff = 0.5 * (1.0 + math.cos(math.pi * decatRatio)) # coeff start at 1 and goes to 0
    return minLr + coEff * (maxLr - minLr)

# Optimizer!
#optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
optim = model.configOptim(weightDecay=0.1, lr=6e-4, device=device)

for i in range(maxSteps):
    t0 = time.time()


    optim.zero_grad()
    lossAccum = 0.0
    for microStep in range(gradAccumStep):
        x, y = trainLoader.nextBatch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we've to scale the loss to account for grad accum
        # Cos the grad just add on each succesiv backward()
        # addition of grad correspond to a sum in the objectiv, but
        # instead of a SUM we want MEAN, scale the loss here so it comes out right

        loss = loss / gradAccumStep
        lossAccum += loss.detach()
        loss.backward()

    #clip the global norm of the grad at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = getLearningRate(i)
    for paraGroup in optim.param_groups:
        paraGroup['lr'] = lr
    optim.step()
    torch.cuda.synchronize()

    t1 = time.time()
    dt = (t1 - t0)*1000 #time dif in mili second

    tokensProcessd = trainLoader.B * trainLoader.T * gradAccumStep
    tokPerSec = tokensProcessd / dt
    print(f'step {i} |, loss:{lossAccum.item():.6f} |, lr:{lr:.4e} norm:{norm:.4f} | dt:{dt:.2f}ms |, tok/sec:{tokensPerSec}')



