import json
from flask_server.university.nlp_utils import tokenize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from neural_net import NeuralNet
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# Load intents with UTF-8 encoding
with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

# Preprocess data
all_words = []
tags = [intent["tag"] for intent in intents["intents"]]
xy = []
puncts = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, intent["tag"]))

# Filter rare words
all_words = [stem(w.lower()) for w in all_words if w not in puncts]
word_counts = Counter(all_words)
all_words = [w for w in all_words if word_counts[w] >= 1]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


# Improved augmentation
def augment_pattern(pattern):
    words = pattern.copy()
    stemmed_words = [stem(w) for w in words]
    key_words = set(stemmed_words).intersection(set(all_words))
    filler_words = ["please", "kindly", "tell", "me", "more", "about"]

    if random.random() > 0.5 and len(words) > 1:
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, random.choice(filler_words))
    if random.random() > 0.5:
        words = [random.choice(["can", "could", "would"]) + " you"] + words
    return words


# Create training data with augmentation
X_train = []
Y_train = []

for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

    for _ in range(2):
        aug_pattern = augment_pattern(pattern_sentence)
        bag = bag_of_words(aug_pattern, all_words)
        X_train.append(bag)
        Y_train.append(label)

# Split into train/validation sets
X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)
X_train = torch.FloatTensor(np.array(X_train_np))  # Fix tensor warning
Y_train = torch.LongTensor(np.array(Y_train_np))
X_val = torch.FloatTensor(np.array(X_val_np))
Y_val = torch.LongTensor(np.array(Y_val_np))


# Define dataset
class ChatDataSet(Dataset):
    def __init__(self, X, Y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = Y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Create data loaders
train_dataset = ChatDataSet(X_train, Y_train)
val_dataset = ChatDataSet(X_val, Y_val)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

# Model parameters
input_size = len(X_train[0])
hidden_size = max(16, len(tags) * 2)  # Increased to 30 for 15 tags
output_size = len(tags)

# Setup model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Training loop
best_val_loss = float("inf")
patience = 20
patience_counter = 0

for epoch in range(1000):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for words, labels in train_loader:
        words, labels = words.to(device), labels.to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for words, labels in val_loader:
            words, labels = words.to(device), labels.to(device)
            outputs = model(words)
            val_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch+1}/1000, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        data = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": hidden_size,
            "all_words": all_words,
            "tags": tags,
        }
        torch.save(data, "data.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

print("Training complete. Best model saved to data.pth")
