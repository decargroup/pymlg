from pymlg.cpp import SO3, SE3, SE23, SL3
from pymlg.numpy import SO3 as SO3np
from pymlg.numpy import SE3 as SE3np
from pymlg.numpy import SE23 as SE23np
from pymlg.numpy import SL3 as SL3np
import pytest

import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.__str__())
from standard_tests import StandardTests
from standard_tests import CrossValidation


@pytest.mark.parametrize("G", [SO3, SE3, SE23, SL3])
class TestStandardCpp(StandardTests):
    pass

@pytest.mark.parametrize("G1, G2", [(SO3, SO3np), (SE3, SE3np), (SE23, SE23np), (SL3, SL3np)])
class TestValidationNumpyCpp(CrossValidation):
    pass


if __name__ == "__main__":
    # For debugging purposes
    test = TestValidationNumpyCpp()
    # test.do_tests(SO3)
    test.test_left_jacobian(SL3, SL3np)
    # test.test_odot_wedge(SO3)
