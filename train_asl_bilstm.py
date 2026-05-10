import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# CONFIG
# ==============================
SEQUENCE_LENGTH = 30
DATASET_DIR = "asl_dataset"
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
PATIENCE = 10  # for early stopping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# ==============================
# Dataset with Train/Test Split
# ==============================
class ASLDataset(Dataset):
    def __init__(self, root, split="train", test_size=0.2, random_state=42):
        self.samples = []
        self.labels = sorted(os.listdir(root))
        self.label_map = {l: i for i, l in enumerate(self.labels)}

        all_samples = []
        for label in self.labels:
            for file in os.listdir(os.path.join(root, label)):
                all_samples.append((os.path.join(root, label, file), self.label_map[label]))

        train_samples, test_samples = train_test_split(
            all_samples,
            test_size=test_size,
            random_state=random_state,
            stratify=[l for _, l in all_samples]
        )

        self.samples = train_samples if split == "train" else test_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)
        return torch.tensor(x, dtype=torch.float32), label

# ==============================
# BiLSTM Model
# ==============================
class ASLBiLSTM(nn.Module):
    def __init__(self, input_size=84, hidden_size=128, num_classes=4, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.layernorm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        out, _ = self.lstm(x)           # shape: (batch, seq_len, hidden*2)
        out = out[:, -1, :]              # last timestep
        out = self.layernorm(out)
        out = self.dropout(out)
        return self.fc(out)

# ==============================
# Load Dataset
# ==============================
train_dataset = ASLDataset(DATASET_DIR, split="train")
test_dataset = ASLDataset(DATASET_DIR, split="test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# Initialize Model
# ==============================
model = ASLBiLSTM(num_classes=len(train_dataset.labels)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
use_amp = torch.cuda.is_available()

if use_amp:
    scaler = torch.amp.GradScaler(device_type="cuda") # mixed precision

# ==============================
# Evaluation Function
# ==============================
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(x)
            else:
                outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, precision, recall, f1, y_true, y_pred

# ==============================
# Training Loop with Early Stopping
# ==============================
best_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    # Evaluate on test set
    acc, precision, recall, f1, _, _ = evaluate(model, test_loader)

    if PRINT_EVERY:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Test Acc: {acc:.4f} | F1: {f1:.4f}")

    # Early stopping
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        torch.save({
            "model": model.state_dict(),
            "labels": train_dataset.labels
        }, "asl_bilstm.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

# ==============================
# Confusion Matrix
# ==============================
_, _, _, _, y_true, y_pred = evaluate(model, test_loader)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=train_dataset.labels,
            yticklabels=train_dataset.labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()