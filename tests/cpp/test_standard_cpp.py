from pylie.cpp import SO3
import pytest

import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.__str__())
from standard_tests import StandardTests


@pytest.mark.parametrize("G", [SO3])
class TestStandardCpp(StandardTests):
    pass


if __name__ == "__main__":
    # For debugging purposes
    test = TestStandardCpp()
    test.do_tests(SO3)
    # test.do_tests(SE3)
    # test.test_odot_wedge(SO3)
