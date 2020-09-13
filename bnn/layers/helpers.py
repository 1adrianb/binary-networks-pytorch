from dataclasses import fields
import torch

from .. import BConfig


def copy_paramters(source_mod: torch.nn.Module, target_mod: torch.nn.Module, bconfig: BConfig) -> None:
    attributes = [field.name for field in fields(bconfig)]
    for attribute in attributes:
        attr_obj_source = getattr(source_mod, attribute, None)
        attr_obj_target = getattr(target_mod, attribute, None)
        for name, source_param in attr_obj_source.named_parameters():
            if source_param is not None and attr_obj_target is not None:
                target_param = getattr(attr_obj_target, name, None)
                if target_param is not None:
                    if torch.equal(torch.tensor(target_param.size()), torch.tensor(source_param.size())):
                        target_param.data.copy_(source_param.data)
