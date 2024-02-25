from pymlg.torch import  SO3, SE3, SE23, SO2, SE2
import pytest
import torch

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())

from standard_tests_torch import StandardTestsTorch

@pytest.mark.parametrize("G", [SO3, SE3, SE23])
class TestStandardTorch(StandardTestsTorch):
    pass

if __name__ == "__main__":

    # set pytorch to double precision for testing
    torch.set_default_dtype(torch.float64)

    # For debugging purposes
    test = TestStandardTorch()
    test.do_tests(SO3)
    test.do_tests(SE3)
    test.do_tests(SE23)
    test.do_tests(SO2)
    test.do_tests(SE2)
