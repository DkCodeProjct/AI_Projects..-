
####################
####################
####################
####################
####################
#NOTE
#NOTE
#  So its too big that Fineweb Dataset so i got
#  This wikiText-2 Dataset,
#  and ask gpt to how can i adapt this dataset to follow alone with the lecture
#  So in this code im USING WIKITEXT-2 Dataset
####################
####################
####################
####################
####################
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Set up the local directory for caching data
localDir = "wikitext2_shards"
shardSiz = int(1e6)  # Smaller shard size for Colab, adjust as needed

# Create the cache directory if it doesn't exist
DATA_CATCH_DIR = os.path.join(os.getcwd(), localDir)
os.makedirs(DATA_CATCH_DIR, exist_ok=True)

# Download the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    # Tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # Add the special <|endoftext|> token at the start
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    # Save tokenized data to a .npy file
    np.save(filename, tokens_np)

# Tokenize all documents and write output shards
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shardSiz,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, dataset["text"], chunksize=16):
        if token_count + len(tokens) < shardSiz:
            # Append tokens to the current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shardSiz, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Write the current shard and start a new one
            remainder = shardSiz - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(
                os.path.join(DATA_CATCH_DIR, f"wikitext2_train_{shard_index:06d}.npy"),
                all_tokens_np
            )
            shard_index += 1
            progress_bar = None

            # Populate the next shard with leftovers from the current doc
            all_tokens_np[:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # Write any remaining tokens as the last shard
    if token_count != 0:
        write_datafile(
            os.path.join(DATA_CATCH_DIR, f"wikitext2_train_{shard_index:06d}.npy"),
            all_tokens_np[:token_count]
        )
