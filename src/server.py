# src/server.py
import os
import torch
import flwr as fl
from transformers import DistilBertForSequenceClassification
import argparse
import numpy as np

# Helper to set parameters on a model
def set_parameters_from_ndarrays(model, parameters):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {}
    for k, p in zip(keys, parameters):
        new_state[k] = torch.tensor(p)
    model.load_state_dict(new_state, strict=True)

def get_initial_parameters():
    # initialize from pretrained DistilBERT
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]

def save_aggregated_model(parameters, save_dir="models/aggregated_model"):
    os.makedirs(save_dir, exist_ok=True)
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    # set params
    set_parameters_from_ndarrays(model, parameters)
    model.save_pretrained(save_dir)
    print(f"Saved aggregated model to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3, help="number of federated rounds")
    parser.add_argument("--num_clients_total", type=int, default=3)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # all clients participate
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients_total,
        min_evaluate_clients=args.num_clients_total,
        min_available_clients=args.num_clients_total,
        initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters()),
    )

    print("Starting Flower server...")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # After server finishes, the strategy should have aggregated parameters saved by Flower's internals,
    # but we also may want to save the last global model manually if we had access to it.
    # Note: fl.server.start_server is blocking and returns after completion.
    # There is not a straightforward returned 'parameters' object here in this simple setup,
    # so as a simple approach we instruct users to fetch the aggregated model from clients or use callbacks.
    print("Server finished.")
