from pylie import SE23 as G
import common


def test_wedge_vee():
    common._test_wedge_vee(G)


def test_exp():
    common._test_exp(G)


def test_log():
    common._test_log(G)


def test_exp_log_inverse():
    common._test_exp_log_inverse(G)


def test_capital_exp_log_inverse():
    common._test_capital_exp_log_inverse(G)


def test_adjoint_identity():
    common._test_adjoint_identity(G)


if __name__ == "__main__":
    test_wedge_vee()
