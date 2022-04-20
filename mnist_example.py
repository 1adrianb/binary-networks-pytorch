from models.mnist import MnistNet
from datasets.datasets import get_mnist
from bnn import BConfig, prepare_binary_model, Identity
from trainers.metrics import get_metrics_and_loss


from run_experiment import run_experiment


from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    TanhBinarizer,
    NoisyTanhBinarizer,
)

model = MnistNet()

bconfig = BConfig(
    activation_pre_process=NoisyTanhBinarizer,
    activation_post_process=Identity,
    weight_pre_process=NoisyTanhBinarizer,
)
# first and last layer will be kept FP32
# model = prepare_binary_model(
#     model,
#     bconfig,
#     custom_config_layers_name={
#         "move1_c": BConfig(),
#         "move1_a": BConfig(),
#         "move2_c": BConfig(),
#         "move2_a": BConfig(),
#     },
# )
# print(model)

target = "label"
train_loader, test_loader = get_mnist()
loaders = {"train": train_loader, "validation": test_loader}
run_experiment(model, loaders, "MNIST_FP", target)
