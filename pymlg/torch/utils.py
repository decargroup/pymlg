import torch

def is_close(t1 : torch.Tensor, t2 : torch.Tensor, tol=1e-6):
    """
    Check element-wise if two tensors are close within some tolerance.
    Either tensor can be replaced by a scalar.
    """
    return (t1 - t2).abs_().lt(tol)

def batch_vector(N, v : torch.Tensor):
    """
    Generated a batched set of the specified vector
    """

    return v.repeat(N, 1, 1)

def batch_eye(N, n, m, dtype = torch.float32):
    """
    Generate a batched set of identity matricies by using torch.repeat()
    """

    b = torch.eye(n, m, dtype=dtype)

    return b.repeat(N, 1, 1)