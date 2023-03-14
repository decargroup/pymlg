import torch

def is_close(t1, t2, tol=1e-6):
    """
    Check element-wise if two tensors are close within some tolerance.
    Either tensor can be replaced by a scalar.
    """
    return torch.all((t1 - t2).abs_().lt(tol))

def batch_eye(N, n, m):
    """
    Generate a batched set of identity matricies by using torch.repeat()
    """

    b = torch.eye(n, m)

    return b.repeat(N, 1, 1)