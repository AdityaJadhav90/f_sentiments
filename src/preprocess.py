# src/preprocess.py

import os
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import pandas as pd

# Create output folder if not exists
os.makedirs("data/imdb_processed", exist_ok=True)

# 1️⃣ Load IMDB dataset
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

train_data = dataset["train"]
test_data = dataset["test"]

# 2️⃣ Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 3️⃣ Convert to Pandas for easy splitting
df = pd.DataFrame({
    "text": train_data["text"],
    "label": train_data["label"]
})

# 4️⃣ Split into client datasets (e.g., 3 clients)
client_1, temp = train_test_split(df, test_size=0.66, random_state=42)
client_2, client_3 = train_test_split(temp, test_size=0.5, random_state=42)

clients = {
    "client_1.csv": client_1,
    "client_2.csv": client_2,
    "client_3.csv": client_3
}

# 5️⃣ Save each client’s data
for name, data in clients.items():
    path = os.path.join("data/imdb_processed", name)
    data.to_csv(path, index=False)
    print(f"Saved {name} with {len(data)} samples")

# 6️⃣ Save test data for later evaluation
test_df = pd.DataFrame({
    "text": test_data["text"],
    "label": test_data["label"]
})
test_df.to_csv("data/imdb_processed/test.csv", index=False)

print("✅ IMDB preprocessing complete!")
