# run_federated.py
import subprocess
import time
import sys
import os

NUM_CLIENTS = 3
SERVER_ADDR = "127.0.0.1:8080"

# Start server
server_proc = subprocess.Popen([sys.executable, "src/server.py", "--rounds", "3", "--num_clients_total", str(NUM_CLIENTS)], stdout=sys.stdout, stderr=sys.stderr)
time.sleep(2)  # give server time to start

# Start clients
clients = []
for i in range(1, NUM_CLIENTS + 1):
    cid = f"client_{i}"
    p = subprocess.Popen([sys.executable, "src/client.py", "--cid", cid, "--server_address", SERVER_ADDR], stdout=sys.stdout, stderr=sys.stderr)
    clients.append(p)
    time.sleep(1)

# Wait for clients to finish
for p in clients:
    p.wait()

# Once clients finish, terminate server
server_proc.terminate()
print("Federated run complete.")
