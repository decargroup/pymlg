from pymlg.torch import  SO3, SE3, SE23, SO2, SE2
import pytest
import torch

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())

from standard_tests_torch import StandardTestsTorch

@pytest.mark.parametrize("G", [SO3, SE3, SE23, SO2, SE2])
# @pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize('device', [
'cpu',
pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA'))
])
class TestStandardTorch(StandardTestsTorch):
    pass

if __name__ == "__main__":

    # set pytorch to double precision for testing
    torch.set_default_dtype(torch.float64)

    # Perform tests on CPU
    test = TestStandardTorch()
    test.do_tests(SO3, device='cpu')
    test.do_tests(SE3, device='cpu')
    test.do_tests(SE23, device='cpu')
    test.do_tests(SO2, device='cpu')
    test.do_tests(SE2, device='cpu')

    # if CUDA is available, perform tests on GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        test.do_tests(SO3, device='cuda')
        test.do_tests(SE3, device='cuda')
        test.do_tests(SE23, device='cuda')
        test.do_tests(SO2, device='cuda')
        test.do_tests(SE2, device='cuda')
