import yaml
import re
from easydict import EasyDict as edict
from typing import List, Callable, Dict

from . import BConfig, prepare_binary_model
from .ops import *


def _option_builder_helper(partial_config: Dict[str, str]) -> str:
    if 'args' in partial_config:
        start_string = '.with_args('
        end_string = ')'
        content = ''
        for k, v in partial_config.args.items():
            content += '{}={},'.format(k, v)
        final_string = start_string + content + end_string

        return final_string
    return ''


class BinaryChef(object):
    r"""Converts a given model according to the configutation and steps defined in an YAML file.ut
    Examples::
        >>> bc = BinaryChef('config.yaml')
        >>> for i in range(len(bc)):
        >>>     model = bc.next(model)
        >>>     # do training logic for desired number of epochs
    Args:
        config: path to a valid yaml file containing the steps
        user_modules: list containing custom user defined binarizers
    """

    def __init__(self, config: str, user_modules: List[Callable[..., nn.Module]] = []) -> None:
        with open(config) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            config = [edict(config[k]) for k in config.keys()]
        self.config = config
        self.current_step = 0
        # inject the received functions into the namespace
        for user_module in user_modules:
            globals()[user_module.__name__] = user_module

    def __len__(self) -> int:
        return len(self.config)

    def get_num_steps(self) -> int:
        return len(self)

    def run_step(self, model: nn.Module, step: int) -> nn.Module:
        assert len(self) > step
        step_config = self.config[step]

        # Ignore certain white listed layers
        print(step_config, type(step_config))
        ignore_layer_names = step_config.ignore_layer_names if 'ignore_layer_names' in step_config else []

        # prepare model
        bconfig = BConfig(
            activation_pre_process=eval(
                step_config.pre_activation.name
                + _option_builder_helper(
                    step_config.pre_activation)),
            activation_post_process=eval(
                step_config.post_activation.name
                + _option_builder_helper(
                    step_config.post_activation)),
            weight_pre_process=eval(
                step_config.weight.name
                + _option_builder_helper(
                    step_config.weight)))
        bmodel = prepare_binary_model(model, bconfig=bconfig, ignore_layers_name=ignore_layer_names)

        return bmodel

    def next(self, model: nn.Module) -> nn.Module:
        self.current_step += 1
        return self.run_step(model, self.current_step - 1)
