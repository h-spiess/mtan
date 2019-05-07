import torch


def inspect_gpu_tensors():
    import gc
    l = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                l.append((type(obj), obj.size()))
        except:
            pass
    return l
