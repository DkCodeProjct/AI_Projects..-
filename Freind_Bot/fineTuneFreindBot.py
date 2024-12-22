
from transformers import AutoTokenizer
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from freind_bot import GPTLanguageModel
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

batchsiz = 64
blocksiz = 128
epochs = 40
lr = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

checkPntPath = "/kaggle/working/FreindBotModelTrainFinl.pth"
model = GPTLanguageModel()
model.state_dict(torch.load(checkPntPath, map_location=device)["model_state_dict"])
model.to(device)

with open("/kaggle/input/friend-bot/fine_tune_dialogTXT.csv", 'r', encoding="utf-8") as file:
    fineTuneTxt = file.read()

def enc(txt, toknizer):
    tokens = toknizer(txt, return_tensors="pt", truncation=True, padding=False)["input_ids"]
    return tokens.flatten()

fineTunedData = enc(fineTuneTxt, tokenizer)

n = int(0.9*len(fineTunedData))
trainData = fineTunedData[:n]
valData = fineTunedData[n:]


def get_batch(split, block_size=128, batch_size=32):
    dataset = trainData if split == "train" else valData
    ix = torch.randint(0, len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
optim = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(epochs):
    model.train()
    for _ in range(len(trainData) // batchsiz):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    model.eval()
    valLoss = 0
    for _ in range(len(valData) // batchsiz):
        xb, yb = get_batch("val")
        _, loss = model(xb, yb)
        valLoss += loss.item()
    valLoss /= (len(valData) // batchsiz)
    print(f"step {i+1} | val loss {valLoss:.4f} ")


# Save the fine-tuned model
torch.save(model.state_dict(), "freindBotModelFineTuned.pth")
print("Fine-tuning complete. Model saved as freindBotModelFineTuned.pth")


##################
################
##############   Chat With The Model
################
##################


def chatWithTheModel(prompt, model, tokenizer, maxNewToken=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        genTxt = model.genarate(input_ids, maxNewToken, tokenizer)

    return genTxt 

while True:
    prompt = input("YOU: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    response = chatWithTheModel(prompt, model, tokenizer)
    print(f"Lacan: {response}")

