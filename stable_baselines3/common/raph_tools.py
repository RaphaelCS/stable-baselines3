import numpy as np
import torch


def list_dict_numpy_to_tensor(obj):
    if isinstance(obj, list):
        if isinstance(obj[0], int):
            return obj
        for i, elem in enumerate(obj):
            obj[i] = list_dict_numpy_to_tensor(elem)
        return obj
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, int):
                return obj
            break
        for key, value in obj.items():
            obj[key] = list_dict_numpy_to_tensor(value)
        return obj
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj)
    elif isinstance(obj, torch.Tensor):
        return obj
    else:
        raise ValueError("Unknown type %s (obj: %s)" % (str(type(obj)), str(obj)))


def list_dict_tensor_to_device(obj, device):
    if isinstance(obj, list):
        if isinstance(obj[0], int):
            return obj
        for i, elem in enumerate(obj):
            obj[i] = list_dict_tensor_to_device(elem, device)
        return obj
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, int):
                return obj
            break
        for key, value in obj.items():
            obj[key] = list_dict_tensor_to_device(value, device)
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    else:
        raise ValueError("Unknown type %s (obj: %s)" % (str(type(obj)), str(obj)))


def list_dict_tensor_print_device(obj):
    if isinstance(obj, list):
        if isinstance(obj[0], int):
            return
        for elem in obj:
            list_dict_tensor_print_device(elem)
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, int):
                return
            break
        for _, value in obj.items():
            list_dict_tensor_print_device(value)
    elif isinstance(obj, torch.Tensor):
        print(obj.device, end=",")
    else:
        raise ValueError("Unknown type %s (obj: %s)" % (str(type(obj)), str(obj)))
