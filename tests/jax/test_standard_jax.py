
#from jax.config import config
from jax import config 
config.update("jax_enable_x64", True)

from pymlg.jax import  SO3, SE3, SE23, SL3, SO2, SE2
import pytest

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from standard_tests import StandardTests

@pytest.mark.parametrize("G", [SO2, SO3, SE2, SE3, SE23, SL3])
class TestStandardJax(StandardTests):
    pass

if __name__ == "__main__":
    # For debugging purposes
    test = TestStandardJax()
    # test.do_tests(SO3)
    # test.do_tests(SE3)
    test.do_tests(SE2)
