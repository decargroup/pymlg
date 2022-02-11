from pylie import SE2
from common import _test_wedge_vee, _test_exp, _test_exp_log_inverse, _test_log

G = SE2 

def test_wedge_vee():
    _test_wedge_vee(G)

def test_exp():
    _test_exp(G)

def test_log():
    _test_log(G)

def test_exp_log_inverse():
    _test_exp_log_inverse(G)

if __name__ == "__main__":
    test_wedge_vee()