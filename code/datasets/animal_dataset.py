import os
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
])

def main():
    # Load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Filter dataset to only include cat, dog, horse (classes 0, 1, 2)
    selected_classes = [3, 5, 7]  # class indices for cat, dog, horse
    label_mapping = {3: 0, 5: 1, 7: 2}

    train_idx = [i for i, label in enumerate(train_dataset.targets) if label in selected_classes]
    test_idx = [i for i, label in enumerate(test_dataset.targets) if label in selected_classes]

    train_dataset.targets = [label_mapping[train_dataset.targets[i]] for i in train_idx]
    test_dataset.targets = [label_mapping[test_dataset.targets[i]] for i in test_idx]

    train_dataset.data = train_dataset.data[train_idx]
    test_dataset.data = test_dataset.data[test_idx]

    # Add random images from other classes to label as class 3 (for other animals)
    other_classes = [i for i in range(10) if i not in selected_classes]  # exclude cat, dog, horse
    random_train_idx = [i for i, label in enumerate(train_dataset.targets) if label in other_classes]
    random_test_idx = [i for i, label in enumerate(test_dataset.targets) if label in other_classes]

    # Randomly sample some images for class 3
    train_samples = 5000  # Sample 5000 images for trainset
    test_samples = 1000 # Sample 1000 images for testset
    random_train_samples = random.sample(random_train_idx, train_samples)
    random_test_samples = random.sample(random_test_idx, test_samples)

    # Assign them to class 3, which represents anything not cat, dog or horse
    train_dataset.targets.extend([3] * len(random_train_samples))
    test_dataset.targets.extend([3] * len(random_test_samples))

    train_dataset.data = np.concatenate((train_dataset.data, train_dataset.data[random_train_samples]))
    test_dataset.data = np.concatenate((test_dataset.data, test_dataset.data[random_test_samples]))

    torch.save(train_dataset, '\'filtered_train_dataset.pth')
    torch.save(test_dataset, '\'filtered_test_dataset.pth')
    print("Datasets saved locally.")
if __name__ == "__main__":
    main()
