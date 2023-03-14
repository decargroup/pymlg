from pylie.cpp import SO3, SE3, SE23
from pylie.numpy import SO3 as SO3np
from pylie.numpy import SE3 as SE3np
from pylie.numpy import SE23 as SE23np
import pytest

import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.__str__())
from standard_tests import StandardTests
from standard_tests import CrossValidation


@pytest.mark.parametrize("G", [SO3, SE3, SE23])
class TestStandardCpp(StandardTests):
    pass

@pytest.mark.parametrize("G1, G2", [(SO3, SO3np), (SE3, SE3np), (SE23, SE23np)])
class TestValidationNumpyCpp(CrossValidation):
    pass


if __name__ == "__main__":
    # For debugging purposes
    test = TestStandardCpp()
    # test.do_tests(SO3)
    test.do_tests(SE3)
    # test.test_odot_wedge(SO3)
