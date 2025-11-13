import matplotlib.pyplot as plt

# Your experiment results
seq_lengths = [25, 50, 100]
accuracies = [0.7021, 0.7537, 0.7928]

plt.figure(figsize=(8, 5))
plt.plot(seq_lengths, accuracies, marker='o', linewidth=2)

plt.title("Accuracy vs Sequence Length", fontsize=14)
plt.xlabel("Sequence Length", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)

plt.xticks(seq_lengths)
plt.ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)

# Save the plot
plt.savefig("accuracy_vs_seq_length.png", dpi=200, bbox_inches="tight")

# Show the plot
plt.show()
