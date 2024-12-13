import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from transformers import GPT2Config, GPT2LMHeadModel

# Load preprocessed data
X = np.load("processed_data/X.npy")
y = np.load("processed_data/y.npy")
unique_tokens = np.load("processed_data/unique_tokens.npy", allow_pickle=True)
seq_attack_arr = np.load("processed_data/seq_attack_arr.npy", allow_pickle=True)
seq_index = np.load("processed_data/seq_index.npy")
class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = FlowDataset(X, y)
batch_size = 512
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
class ProtoBytesGPT2(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(ProtoBytesGPT2, self).__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size + 1,
            n_embd=embed_dim,
            n_layer=4,  # Number of transformer layers
            n_head=4    # Number of attention heads
        )
        self.gpt2 = GPT2LMHeadModel(self.config)

    def forward(self, x):
        # GPT-2 forward pass
        outputs = self.gpt2(input_ids=x, labels=x)
        loss, logits = outputs.loss, outputs.logits
        return loss, logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProtoBytesGPT2(vocab_size=len(unique_tokens)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
epochs = 15
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)

        optimizer.zero_grad()
        loss, _ = model(batch_x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), "models/protobytes_gpt2.pt")
model.eval()
test_dataset = FlowDataset(X, y)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

all_preds = []
all_y = []
logloss_list = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        _, logits = model(batch_x)
        probs = torch.softmax(logits, dim=-1)

        # Compute log loss for each token
        for seq_idx, seq_probs in enumerate(probs):
            for i in range(1, seq_probs.size(0)):  # Ignore the first token
                correct_token_prob = seq_probs[i, batch_x[seq_idx, i].item()].item()
                logloss_list.append(-math.log(correct_token_prob + 1e-10))
key_ll = zip(seq_index, logloss_list, seq_attack_arr)
dictionary = dict()
for (key, ll, aa) in key_ll:
    current_value = dictionary.get(str(key), ([], []))
    dictionary[str(key)] = (current_value[0] + [ll], current_value[1] + [aa])

agg_ll = []
agg_bad = []
for key, val in dictionary.items():
    bad = str(np.mean([v == "Attack" for v in val[1]]) > 0.)
    score = np.max(val[0])
    agg_bad.append(bad)
    agg_ll.append(score)

fpr, tpr, thresholds = roc_curve([x == "True" for x in agg_bad], agg_ll, pos_label=True)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f"ROC curve (AUC = {auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ProtoBytes with GPT-2')
plt.legend(loc="lower right")
plt.savefig("graphics/protobytes_gpt2.pdf", format="pdf")
plt.show()
