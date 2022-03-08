from pylie import SE3
import numpy as np
from common import _test_wedge_vee, _test_exp, _test_exp_log_inverse, _test_log

G = SE3


def test_wedge_vee():
    _test_wedge_vee(G)


def test_exp():
    _test_exp(G)


def test_log():
    _test_log(G)


def test_exp_log_inverse():
    _test_exp_log_inverse(G)
