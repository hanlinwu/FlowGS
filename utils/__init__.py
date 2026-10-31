import importlib
from functools import partial

import torch


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        if config in ("__is_first_stage__", "__is_unconditional__"):
            return None
        raise KeyError("Expected target to instantiate.")
    params = config.get("params", {})
    return get_obj_from_str(config["target"])(**params)


def load_partial_from_config(config):
    if "target" not in config:
        raise KeyError("Expected target to instantiate partial.")
    params = config.get("params", {})
    return partial(get_obj_from_str(config["target"]), **params)


def exists(value):
    return value is not None


def default(value, fallback):
    if exists(value):
        return value
    return fallback() if callable(fallback) else fallback


def disabled_train(self, mode=True):
    return self


def ismap(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.ndim == 4 and tensor.shape[1] > 3


def isimage(tensor):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.ndim == 4
        and tensor.shape[1] in (1, 3)
    )


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total = sum(parameter.numel() for parameter in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total * 1e-6:.2f} M params.")
    return total
