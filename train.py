import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pickle
import os

from dataset import CaptionDataset
from vocab import build_vocab
from utils_data import load_captions

# -----------------------------
# 1. LOAD DATA
# -----------------------------
mapping, captions = load_captions("data/captions.txt")
print("Total captions:", len(captions))
print("Total images:", len(mapping))
# -----------------------------
# 2. BUILD VOCAB
# -----------------------------
word2idx, idx2word = build_vocab(captions)

os.makedirs("model", exist_ok=True)

with open("model/vocab.pkl", "wb") as f:
    pickle.dump((word2idx, idx2word), f)

# -----------------------------
# 3. DATASET
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CaptionDataset("data/images", mapping, word2idx, transform)
from dataset import collate_fn

loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# -----------------------------
# 4. MODEL
# -----------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        return self.fc(features)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(inputs)
        return self.fc(outputs)


class Model(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)


# -----------------------------
# 5. TRAINING SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(256, 512, len(word2idx)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# -----------------------------
# 6. TRAIN LOOP
# -----------------------------
EPOCHS = 10   # keep small for first run

for epoch in range(EPOCHS):
    for images, captions in loader:

        images = images.to(device)
        captions = captions.to(device)

        outputs = model(images, captions[:, :-1])
        outputs = outputs[:, 1:, :]


        loss = criterion(
            outputs.reshape(-1, len(word2idx)),
            captions[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# -----------------------------
# 7. SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "model/model.pth")

print("✅ Training complete! Model saved.")