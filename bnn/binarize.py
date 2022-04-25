from dataclasses import asdict
from .bconfig import BConfig
import re
import copy
import logging
import torch
import torch.nn as nn
from typing import Dict, List

from . import layers as bnn

DEFAULT_MODULE_MAPPING = {
    nn.Linear: bnn.Linear,
    nn.Conv2d: bnn.Conv2d,
    nn.Conv1d: bnn.Conv1d,
}
for k, v in copy.copy(DEFAULT_MODULE_MAPPING).items():
    DEFAULT_MODULE_MAPPING[v] = v


def _get_first_layer(model: nn.Module) -> List[str]:
    for name, module in model.named_modules():
        if type(module) in DEFAULT_MODULE_MAPPING:
            return [name]
    return []


def _get_last_layer(model: nn.Module) -> List[str]:
    for name, module in reversed(list(model.named_modules())):
        if type(module) in DEFAULT_MODULE_MAPPING:
            return [name]
    return []


def _regex_match(
    model: nn.Module, pattern: str, modules_mapping: Dict[nn.Module, nn.Module]
) -> List[str]:
    # Strip first and last character (expected to be $ and $)
    pattern = pattern[1:-1]

    matched_names = []
    pattern = re.compile(pattern)
    for name, module in model.named_modules():
        if type(module) in modules_mapping:
            if pattern.search(name) is not None:
                matched_names.append(name)
    return matched_names


_KNOWN_SPECIAL_WORDS = {"_last_": _get_first_layer, "_first_": _get_last_layer}


def get_unique_devices_(module: nn.Module) -> List[torch.device]:
    return {p.device for p in module.parameters()} | {
        p.device for p in module.buffers()
    }


def get_modules_to_binarize(
    model: nn.Module,
    bconfig: BConfig,
    modules_mapping: Dict[nn.Module, nn.Module] = None,
    custom_config_layers_name: Dict[str, BConfig] = {},
    ignore_layers_name: List[str] = [],
) -> Dict[str, nn.Module]:
    if modules_mapping is None:
        modules_mapping = DEFAULT_MODULE_MAPPING

    # Parse special cases
    processed_ignore_layer_names = []
    for name in ignore_layers_name:
        if name in _KNOWN_SPECIAL_WORDS.keys():
            processed_ignore_layer_names += _KNOWN_SPECIAL_WORDS[name](model)
        elif name[0] == "$" and name[-1] == "$":
            processed_ignore_layer_names += _regex_match(
                model, name, modules_mapping
            )
        else:
            processed_ignore_layer_names.append(name)

    modules_to_replace = {}
    for name, module in model.named_modules():
        if type(module) in modules_mapping:
            if name in processed_ignore_layer_names:
                continue

            layer_config = copy.copy(bconfig)
            # Use layer specific customization
            if name in custom_config_layers_name:
                print(name)
                for k, v in asdict(custom_config_layers_name[name]).items():
                    setattr(layer_config, k, v)

            # respect device affinity when swapping modules
            devices = get_unique_devices_(module)
            assert len(devices) <= 1, (
                "swap_module only works with cpu or single-device CUDA modules, "
                "but got devices {}".format(devices)
            )
            device = next(iter(devices)) if len(devices) > 0 else None

            modules_to_replace[name] = modules_mapping[
                type(module)
            ].from_module(module, layer_config)
            if device:
                modules_to_replace[name].to(device)
        elif name in custom_config_layers_name:
            logging.warning(
                "Module named {} defined in the configuration was not found.".format(
                    name
                )
            )
    return modules_to_replace


def swap_modules_by_name(
    model: nn.Module,
    modules_to_replace: Dict[str, nn.Module],
    modules_mapping: Dict[nn.Module, nn.Module] = None,
) -> nn.Module:
    if modules_mapping is None:
        modules_mapping = DEFAULT_MODULE_MAPPING

    def _swap_module(module: nn.Module):
        for name, child in module.named_children():
            if type(child) in modules_mapping:
                for n, m in model.named_modules():
                    if child is m:
                        if n in modules_to_replace:
                            f = modules_to_replace.pop(n)
                            setattr(module, name, f)
                        break
            else:
                _swap_module(child)

    if len(list(model.named_children())) == 0:
        if (
            type(model) in modules_mapping
            and len(modules_to_replace.keys()) == 1
        ):
            model = modules_to_replace[list(modules_to_replace.keys())[0]]
    else:
        _swap_module(model)
    return model


def prepare_binary_model(
    model: nn.Module,
    bconfig: BConfig,
    modules_mapping: Dict[nn.Module, nn.Module] = None,
    custom_config_layers_name: Dict[str, BConfig] = {},
    ignore_layers_name: List[str] = [],
) -> nn.Module:
    modules_to_replace = get_modules_to_binarize(
        model,
        bconfig,
        modules_mapping,
        custom_config_layers_name,
        ignore_layers_name,
    )
    model = swap_modules_by_name(model, modules_to_replace, modules_mapping)

    return model
