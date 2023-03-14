import torch

def is_close(t1, t2, tol=1e-6):
    """
    Check element-wise if two tensors are close within some tolerance.
    Either tensor can be replaced by a scalar.
    """
    return torch.all((t1 - t2).abs_().lt(tol))

# left jacobian helpers

def A_lj(t_norm, small=True):
    t2 = t_norm**2
    if small:
        return (1.0 / 2.0) * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0)))
    else:
        return (1 - torch.cos(t_norm)) / (t_norm**2)


def B_lj(t_norm, small=True):
    t2 = t_norm**2
    if small:
        return (1.0 / 6.0) * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0)))
    else:
        return (t_norm - torch.sin(t_norm)) / (t_norm**3)