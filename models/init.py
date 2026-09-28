#Loading the datasets
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

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
