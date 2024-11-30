import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
# ===============================


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.nEmb % config.nHead == 0
        
        # K, Q, V Projections for all heads, but in batch
        self.c_attn = nn.Linear(config.nEmb, 3 * config.nEmb)

        # output projection
        self.c_proj = nn.Linear(config.nEmb, config.nEmb)

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
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.siz(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        y =  attn @ v # (B, nh, T, T) x (B, nh, T, hs) ==> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.c_proj(y)

        return y    



class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.nEmb, 4 * config.nEmb) # fuly conected
        self.gelu = nn.GELU(approximate='tanh') # Read the paper for Aproxiation
        self.c_proj = nn.Linear(4 * config.nEmb, config.nEmb)

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
        

        ####
        # Loading para from the hugging face to our code, and init GPT class with those para
        ####

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
    
model = GPT.from_pretrained('gpt2')
print("Didnt Crash yay... [wft]")

