from .so3 import SO3
from .se3 import SE3
from .base import MatrixLieGroup
from jax.config import config
config.update("jax_enable_x64", True)

SO3.run_everything_once()