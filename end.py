# ================================
# Week 3: Sentiment-Aware LSTM
# ================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Dataset Class (Sliding Window)
# -------------------------------

class MarketDataset(Dataset):
    def __init__(self, features, targets, sequence_length=60):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y


# -------------------------------
# 2. Sentiment-Aware LSTM Model
# -------------------------------

class SentimentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]           # last time step
        out = self.fc(out)
        return out


# -------------------------------
# 3. Hyperparameters
# -------------------------------

INPUT_DIM = 7      # Returns, Volatility, RSI, MACD, Volume, Sentiment, etc.
HIDDEN_DIM = 64
NUM_LAYERS = 2
SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# 4. Dummy Data (Replace with real)
# -------------------------------

np.random.seed(42)
total_days = 1200

features = np.random.randn(total_days, INPUT_DIM)

time = np.linspace(0, 50, total_days)
targets = np.sin(time) + 0.1 * np.random.randn(total_days)

train_size = 1000
train_features = features[:train_size]
test_features = features[train_size:]

train_targets = targets[:train_size]
test_targets = targets[train_size:]


# -------------------------------
# 5. DataLoaders
# -------------------------------

train_dataset = MarketDataset(train_features, train_targets, SEQ_LEN)
test_dataset = MarketDataset(test_features, test_targets, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# -------------------------------
# 6. Model, Loss, Optimizer
# -------------------------------

model = SentimentLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# -------------------------------
# 7. Training Loop
# -------------------------------

print("Training started...\n")

model.train()
for epoch in range(EPOCHS):
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(X).squeeze()
        loss = criterion(output, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.6f}")


# -------------------------------
# 8. Evaluation + Visualization
# -------------------------------

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        pred = model(X)
        predictions.append(pred.item())
        actuals.append(y.item())

plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Actual", alpha=0.7)
plt.plot(predictions, label="Predicted", linestyle="--")
plt.title("LSTM: Actual vs Predicted")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------------
# 9. Trading Strategy Logic
# -------------------------------

def get_trade_signal(lstm_prediction, sentiment_score, threshold=0.001):
    if lstm_prediction > threshold and sentiment_score > 0.2:
        return "BUY"
    elif lstm_prediction < -threshold and sentiment_score < -0.2:
        return "SELL"
    else:
        return "HOLD"


# Example usage
example_prediction = 0.015
example_sentiment = 0.8

signal = get_trade_signal(example_prediction, example_sentiment)
print("\nTrade Signal:", signal)
