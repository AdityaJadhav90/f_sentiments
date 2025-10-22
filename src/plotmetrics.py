import matplotlib.pyplot as plt
import os

# Create folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

rounds = [1, 2, 3]
loss_values = [0.2633, 0.2368, 0.2290]
accuracy_values = [0.88, 0.90, 0.91]

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(rounds, loss_values, marker='o', color='red', label='Loss')
plt.title('Federated Training Loss per Round')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.xticks(rounds)
plt.grid(True)
plt.legend()
plt.savefig('plots/loss_per_round.png')
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.plot(rounds, accuracy_values, marker='o', color='green', label='Accuracy')
plt.title('Federated Training Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.xticks(rounds)
plt.grid(True)
plt.legend()
plt.savefig('plots/accuracy_per_round.png')
plt.show()
