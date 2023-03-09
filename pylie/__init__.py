from .numpy.so3 import SO3
from .numpy.se3 import SE3
from .numpy.so2 import SO2
from .numpy.se2 import SE2
from .numpy.se23 import SE23
from .numpy.sl3 import SL3
from .numpy.base import MatrixLieGroup

try:
    from . import torch
except ImportError:
    pass



def set_jax_usage(use_jax):
    """
    Set whether to use Jax's JIT compilation internally.

    Parameters
    ----------
    use_jax: bool
        Whether to use Jax's JIT compilation internally.

    """

    try:
        from . import jax

        jax_available = True
    except ImportError:
        jax_available = False

    MatrixLieGroup.use_jax = use_jax
    if use_jax:
        if not jax_available:
            raise RuntimeError(
                "Jax is not available. Please install by following instructions"
                + " at https://jax.readthedocs.io/en/latest/installation.html"
            )
        else:
            SO3.use_jax = True
            SE3.use_jax = True
            SO2.use_jax = True
            SE2.use_jax = True
            SE23.use_jax = True
            SL3.use_jax = True
    else:
        SO3.use_jax = False
        SE3.use_jax = False
        SO2.use_jax = False
        SE2.use_jax = False
        SE23.use_jax = False
        SL3.use_jax = False
