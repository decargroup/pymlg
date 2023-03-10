from pylie import  SO2, SO3, SE2, SE3, SE23, SL3
import pytest

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())

from standard_tests import StandardTests

@pytest.mark.parametrize("G", [SO2, SO3, SE2, SE3, SE23, SL3])
class TestStandardNumpy(StandardTests):
    pass


if __name__ == "__main__":
    # For debugging purposes
    test = TestStandardNumpy()
    test.do_tests(SO2)
    test.do_tests(SO3)
    test.do_tests(SE2)
    test.do_tests(SE3)
    test.do_tests(SE23)
    test.do_tests(SL3)
