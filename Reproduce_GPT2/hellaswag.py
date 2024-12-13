"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

####################
####################
####################
####################
####################
#NOTE
#NOTE
# before i use Wikitext-2 dataset instead of fineweb, 
# so i ask GPT to adapt this Hellaswag Code
# To Wikitext-2
####################
####################
####################
####################
####################
import os
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import requests 

# Define paths
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "wikitext")
WIKITEXT_URL = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "val": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def download(split):
    """Download WikiText dataset."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = WIKITEXT_URL[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"wikitext_{split}.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def iterate_examples(split, tokenizer, block_size=512):
    """Yield tokenized chunks for WikiText."""
    download(split)
    data_path = os.path.join(DATA_CACHE_DIR, f"wikitext_{split}.txt")
    with open(data_path, "r") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens) - block_size, block_size):
        input_ids = tokens[i : i + block_size]
        yield torch.tensor(input_ids)

@torch.no_grad()
def evaluate(model_type, device, split="val"):
    """Evaluate the model on WikiText."""
    torch.set_float32_matmul_precision("high")  # use tf32
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    total_loss = 0.0
    num_tokens = 0
    for input_ids in iterate_examples(split, tokenizer):
        input_ids = input_ids.to(device)
        labels = input_ids.clone()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        total_loss += loss.item() * input_ids.size(0)
        num_tokens += input_ids.size(0)

    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(total_loss / num_tokens))
    print(f"{split} Perplexity: {perplexity.item()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
