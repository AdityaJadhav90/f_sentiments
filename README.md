# 🧠 Federated Sentiment Analysis using Flower & DistilBERT

## Overview
This project presents a **Federated Learning (FL)** system for **Sentiment Analysis** using **DistilBERT**, a lightweight transformer model.  
The framework leverages **Flower (FLWR)** for distributed model training and **Flask** for serving the aggregated global model as a REST API.  
This approach enables decentralized training on multiple clients while maintaining data privacy, with the global model continuously improving through federated aggregation.

---

## 🌼 Federated Learning Setup (Flower)

### Architecture
The system comprises three main components:

1. **Flower Server (Aggregator):**  
   Acts as the central coordinator that performs model aggregation using the **Federated Averaging (FedAvg)** algorithm.

2. **Clients:**  
   Each client trains a local instance of DistilBERT using its private dataset (e.g., IMDB reviews, Twitter sentiment, etc.).  
   Clients send only model weight updates—not raw data—to the central server, preserving data privacy.

3. **Global Model:**  
   After several FL rounds, the server aggregates client updates into a single **global model** stored under:
