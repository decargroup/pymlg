from pymlg.torch import  SO3, SE3, SE23
import pytest

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())

from standard_tests import StandardTests

@pytest.mark.parametrize("G", [SO3, SE3, SE23])
class TestStandardTorch(StandardTests):
    pass

if __name__ == "__main__":
    # For debugging purposes
    test = TestStandardTorch()
    test.do_tests(SO3)
    test.do_tests(SE3)
    test.do_tests(SE23)
