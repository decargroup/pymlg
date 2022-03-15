from pylie import SO3
import numpy as np
import common
G = SO3


def test_wedge_vee():
    common._test_wedge_vee(G)


def test_exp():
    common._test_exp(G)


def test_log():
    common._test_log(G)


def test_exp_log_inverse():
    common._test_exp_log_inverse(G)


def test_odot_wedge():
    common._test_odot_wedge(G)