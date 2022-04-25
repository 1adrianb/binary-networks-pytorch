from fastargs import get_current_config
import numpy as np
from models.group_net import resnet18
from datasets.datasets import get_tiny_image_net
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment

from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    TanhBinarizer,
    NoisyTanhBinarizer,
    BasicScaleBinarizer,
)

model = resnet18(num_classes=200)

bconfig = BConfig(
    #activation_pre_process=TanhBinarizer,
    #activation_post_process=Identity,
    #weight_pre_process=TanhBinarizer,
)


model = prepare_binary_model(
    model,
    bconfig,
    # custom_config_layers_name={
    #     "move1_c": BConfig(),
    #     "move1_a": BConfig(),
    #     "move2_c": BConfig(),
    #     "move2_a": BConfig(),
    #},
)

print(model)

target = "label"

run_experiment(model,get_tiny_image_net, target)


# np.save(
#     "./filters_TanhBinarizer_L",
#     model.conv1.weight.data.detach().cpu().numpy(),
# )
