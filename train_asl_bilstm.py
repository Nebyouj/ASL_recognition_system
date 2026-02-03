import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================
# Dataset
# ==============================
class ASLDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.labels = sorted(os.listdir(root))
        self.label_map = {l:i for i,l in enumerate(self.labels)}

        for label in self.labels:
            for file in os.listdir(os.path.join(root, label)):
                self.samples.append(
                    (os.path.join(root, label, file), self.label_map[label])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)
        return torch.tensor(x, dtype=torch.float32), label

# ==============================
# Model
# ==============================
class ASLBiLSTM(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# ==============================
# Training
# ==============================
dataset = ASLDataset("asl_dataset")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ASLBiLSTM(num_classes=len(dataset.labels))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

torch.save({
    "model": model.state_dict(),
    "labels": dataset.labels
}, "asl_bilstm.pth")
