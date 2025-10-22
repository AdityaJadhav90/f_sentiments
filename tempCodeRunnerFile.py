# Import libraries
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# Example input
text = "Hello world!"
inputs = tokenizer(text, return_tensors="pt").to(device)

# Run model
outputs = model(**inputs)

# Print last hidden state shape
print(outputs.last_hidden_state.shape)
