"""
Perform any other group-specific tests that are not part of the standard tests.
"""

from pymlg.torch import SO3 as G
import torch
from random import randrange

def test_quaternion():
    q = torch.Tensor([1, 2, 3, 4]).reshape((1, -1)).repeat(randrange(1, 10), 1)
    q = q / torch.linalg.norm(q, dim=1).unsqueeze(1)
    C = G.from_quat(q, order="wxyz")
    assert torch.allclose(torch.linalg.det(C), torch.ones(C.shape[0]))
    assert torch.allclose((C @ C.transpose(1, 2)), torch.eye(3))

    q_test = G.to_quat(C, order="wxyz")
    assert torch.allclose(q, q_test)

    C_test = G.from_quat(-q, order="wxyz")
    assert torch.allclose(C, C_test)

    # Different order
    C = G.from_quat(q, order="xyzw")
    assert torch.allclose(torch.linalg.det(C), torch.ones(C.shape[0]))
    assert torch.allclose((C @ C.transpose(1, 2)), torch.eye(3))

    q_test = G.to_quat(C, order="xyzw")
    assert torch.allclose(q, q_test)

    C_test = G.from_quat(-q, order="xyzw")
    assert torch.allclose(C, C_test)


if __name__ == "__main__":

    # set pytorch to double precision for testing
    torch.set_default_dtype(torch.float64)

    test_quaternion()
