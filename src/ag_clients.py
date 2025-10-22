import torch
from transformers import DistilBertForSequenceClassification
import os

# Paths to client models
client_paths = [
    "models/client_models/client_1",
    "models/client_models/client_2",
    "models/client_models/client_3"
]

# Load all client models
client_models = []
for path in client_paths:
    model = DistilBertForSequenceClassification.from_pretrained(path)
    client_models.append(model)

# Initialize new global model (same architecture)
global_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Perform FedAvg: average weights of all clients
global_dict = global_model.state_dict()
for key in global_dict.keys():
    # Sum weights from all clients
    global_dict[key] = sum([m.state_dict()[key] for m in client_models]) / len(client_models)

# Load averaged weights into global model
global_model.load_state_dict(global_dict)

# Create folder to save aggregated global model
agg_path = "models/aggregated_model"
os.makedirs(agg_path, exist_ok=True)

# Save the global model
global_model.save_pretrained(agg_path)
print(f"✅ Aggregated global model saved at '{agg_path}'")
