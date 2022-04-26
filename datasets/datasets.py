import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import webdataset as wds


class DictSet(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset
        self.size = len(dataset)

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(int(index))
        return {"model_input": image, "label": label}

    def __len__(self):
        return self.size


def get_mnist(data_path=None, batch_size=-1, workers=-1):
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


def get_tiny_image_net(data_path='/home/dev/data_main/CORESETS/TinyImagenet/tiny-224', batch_size=32, workers=4):

 # Data loading code
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    crop_size, short_size = 224, 256
    train_dataset = DictSet(datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])))

    val_dataset = DictSet(datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(short_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader


def get_tiny_image_net_wds(data_path='/home/dev/data_main/CORESETS/TinyImagenet_wds', batch_size=32, workers=4):

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    # TODO pass correct number of shards
    url = data_path + "/imagenet-train-{000000..000001}.tar"
    train_dataset = DictSet(
        wds.WebDataset(url)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transforms_train, lambda x: x)
        .shuffle(1000)
    )
    # TODO pass correct number of shards
    url = data_path + "/imagenet-val-{000000..000000}.tar"
    val_dataset = DictSet(
        wds.WebDataset(url)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transforms_val, lambda x: x)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader
