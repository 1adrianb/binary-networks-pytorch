import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from datasets.datasets import get_mnist


from models.mnist import MnistNet

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


train_loader, test_loader = get_mnist()


# scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


def weight_norm(weight):
    device = weight.device
    return torch.nn.Parameter(weight / torch.norm(weight)).to(device)


model = MnistNet()

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
