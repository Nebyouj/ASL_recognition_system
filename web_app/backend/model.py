import torch
import torch.nn as nn

class ASLBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    labels = checkpoint["labels"]

    model = ASLBiLSTM(
        input_size=84,
        hidden_size=128,
        num_classes=len(labels)
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, labels
