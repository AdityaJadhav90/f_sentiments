# src/client.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
import flwr as fl

# ---------- Dataset wrapper ----------
class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ---------- Parameter conversion helpers ----------
def get_parameters(model):
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state_dict = {}
    for k, p in zip(keys, parameters):
        new_state_dict[k] = torch.tensor(p)
    model.load_state_dict(new_state_dict, strict=True)

# ---------- Flower NumPyClient ----------
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, train_loader, val_loader, device):
        self.cid = cid
        self.device = device
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model_dir = os.path.join("models", "client_models", cid)
        if os.path.isdir(model_dir):
            # load previous fine-tuned if exists
            try:
                self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            except Exception:
                pass
        self.model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = 1  # local epochs per round
        self.lr = 5e-5

    def get_parameters(self):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # set incoming global params
        set_parameters(self.model, parameters)

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            for batch in self.train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # save local model checkpoint after local training
        save_dir = os.path.join("models", "client_models", self.cid)
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)

        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # set incoming global params
        set_parameters(self.model, parameters)
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                loss_total += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        if total == 0:
            return float(loss_total), 0, {"accuracy": 0.0}
        loss_avg = loss_total / total
        accuracy = correct / total
        return float(loss_avg), total, {"accuracy": float(accuracy)}

# ---------- Utility to load local csv and produce dataloaders ----------
def load_data_for_client(cid, tokenizer, batch_size=8):
    csv_path = os.path.join("data", "imdb_processed", f"{cid}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    # small validation split
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    train_ds = IMDbDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_ds = IMDbDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, required=True, help="client id filename e.g. client_1")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_loader, val_loader = load_data_for_client(args.cid, tokenizer, batch_size=8)

    client = FLClient(args.cid, train_loader, val_loader, device)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
