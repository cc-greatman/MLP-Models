#Loading the datasets
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

train_dataset = datasets.MNIST(
    root='../datasets', train=True, transform=transforms.ToTensor(), download=True
)   

test_dataset = datasets.MNIST(
    root="../datasets", train=False, transform=transforms.ToTensor()
)


# Splitting the Validation Dataset
torch.manual_seed(1)
train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])


# Loading the data with a batch size of 64
train_loader = DataLoader(
    dataset     = train_dataset,
    batch_size  = 64,
    shuffle     = True,
)

val_loader = DataLoader(
    dataset    = val_dataset,
    batch_size = 64,
    shuffle    = False,
)

test_loader = DataLoader(
    dataset    = test_dataset,
    batch_size = 64, 
    shuffle    = False,
)


# Checking Label Distribution in the different subsets
train_counter = Counter()
for images, labels in train_loader:
    train_counter.update(labels.tolist()) # creates a count dictionary and prints it out

print("\nTraining Label Distribution:")
print(sorted(train_counter.items()))

val_counter = Counter()
for images, labels in val_loader:
    val_counter.update(labels.tolist())

print("\nValidation Label Distribution:")
print(sorted(val_counter.items()))

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())

print("\nTest Label Distribution:")
print(sorted(test_counter.items()))


# Initiating a Zero-Rule Classifier
majority_class = test_counter.most_common(1)[0]
print("Majority Class:", majority_class[0])

baseline_acc = majority_class[1] / sum(test_counter.values())
print("Accuracy when always predicting the majority class:")
print(f"{baseline_acc:.2f} ({baseline_acc*100:.2f}%)")

# Visualize the training data images
for images, labels in train_loader:
    break

plt.figure(figsize = (8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(
    images[:64],
    padding = 1,
    pad_value = 1.0,
    normalize = True),
    (1, 2, 0)))
plt.show()

# Implementing the Model
torch.flatten(images, start_dim=1).shape # Batchsize, features

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50,25),
            torch.nn.ReLU(),
            # Output layer
            torch.nn.Linear(25, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits

# Defining the training loop
def compute_accuracy(model, dataloader):

    model = model.eval()

    correct =0.0
    total_examples =0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.inference_mode():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples

# Beginning the training
torch.manual_seed(1)
model = PyTorchMLP(num_features=784, num_classes=10)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

num_epochs = 10

loss_list = []
train_acc_list, val_acc_list = [], []
for epoch in range(num_epochs):

    model = model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not batch_idx % 250:
            ### Logging
            print(
                f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train Loss: {loss:.2f}"
            )

        loss_list.append(loss.item())

    train_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, val_loader)
    print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)