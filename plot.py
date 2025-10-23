# %%
import matplotlib.pyplot as plt

from gnnexp.main import main_mlp, main_transformer

percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

accuracies_mlp = []
accuracies_transformer = []

for p in percentages:
    print(f"Running experiment with {p}% lying...")
    print("MLP Model:")
    acc = main_mlp(perc_lying=p)
    accuracies_mlp.append(acc)

    print("Transformer Model:")
    acc = main_transformer(perc_lying=p)
    accuracies_transformer.append(acc)
# %%
len(accuracies_mlp), len(accuracies_transformer), len(percentages[:5])
# %%
plt.figure(figsize=(10, 6))
plt.plot(percentages[:5], accuracies_mlp, marker='o', label='MLP')
plt.plot(percentages[:5], accuracies_transformer, marker='s', label='Transformer')
plt.title('Model Accuracy vs Percentage of Lying Nodes')
plt.xlabel('Percentage of Lying Nodes')
plt.ylabel('Test Accuracy')
plt.legend()
plt.xticks(percentages[:5])
plt.grid()
plt.savefig('model_accuracy_vs_lying_percentage.png')
plt.show()

print("Experiment completed and plot saved.")

# %%
