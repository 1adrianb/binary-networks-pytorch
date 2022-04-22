import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms


class DictSet(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset
        self.size = dataset.data.shape[0]

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(int(index))
        return {"model_input": image, "label": label}

    def __len__(self):
        return self.size


def get_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = DictSet(
        datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
    )

    dataset2 = DictSet(
        datasets.MNIST("../data", train=False, transform=transform)
    )

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=32)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=32)
    return train_loader, test_loader
