import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data.dataset import Dataset

from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    TanhBinarizer,
    NoisyTanhBinarizer,
)
from bnn import BConfig, prepare_binary_model, Identity


from trainers.metrics import get_metrics_and_loss
from trainers.trainer import Trainer

optimizer = {
    "main": (
        "ADAM",
        {
            "lr": 0.001,
            "weight_decay": 0,
        },
    )
}


TARGET = "label"
criterion, metrics = get_metrics_and_loss(
    "CrossEntropyLoss", ["accuracy", "f1_macro"], TARGET
)

trainer = Trainer(
    criterion,
    metrics,
    optimizers=optimizer,
    phases=["train", "validation"],
    num_epochs=10,
    device=1,
    logger=None,
)


class DictSet(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset
        self.size = dataset.data.shape[0]

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(int(index))
        return {"model_input": image, "label": label}

    def __len__(self):
        return self.size


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


dataset1 = DictSet(
    datasets.MNIST("../data", train=True, download=True, transform=transform)
)

dataset2 = DictSet(datasets.MNIST("../data", train=False, transform=transform))

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=32)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=32)


# scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


def weight_norm(weight):
    device = weight.device
    return torch.nn.Parameter(weight / torch.norm(weight)).to(device)


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(
            torch.zeros(1, out_chn, 1, 1), requires_grad=True
        )

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, 1)
        self.move1_c = LearnableBias(1)
        self.move1_a = LearnableBias(128)
        self.conv2 = nn.Conv2d(128, 64, 3, 1)
        self.move2_c = LearnableBias(128)
        self.move2_a = LearnableBias(64)

        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x = self.move1_c(x)
        x = self.conv1(x)
        # x = self.move1_a(x)
        x = F.relu(x)
        # x = self.move2_c(x)
        x = self.conv2(x)
        # x = self.move2_a(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        # print(self.move2_a.bias)
        output = F.log_softmax(x, dim=1)
        # print(self.conv1.weight)
        return {"preds": output}


model = Net()

bconfig = BConfig(
    activation_pre_process=NoisyTanhBinarizer,
    activation_post_process=Identity,
    weight_pre_process=NoisyTanhBinarizer,
)
# first and last layer will be kept FP32
model = prepare_binary_model(
    model,
    bconfig,
    custom_config_layers_name={
        "move1_c": BConfig(),
        "move1_a": BConfig(),
        "move2_c": BConfig(),
        "move2_a": BConfig(),
    },
)
print(model)

loaders = {"train": train_loader, "validation": test_loader}


trainer.set_model(model, {"main": model.parameters()})
trainer.set_dataloaders(loaders)
trainer.train()
history = trainer.get_history()
