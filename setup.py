# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import os
from pathlib import Path
SETUP_DIRECTORY = Path(__file__).resolve().parent
__version__ = "1.0.0"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    def __str__(self) -> str:
        eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)

        if eigen_include_dir is not None:
            return eigen_include_dir

        target_dir = SETUP_DIRECTORY / self.EIGEN3_DIRNAME
        if target_dir.exists():
            return target_dir.name

        download_target_dir = SETUP_DIRECTORY / "eigen3.zip"
        import zipfile

        import requests

        response = requests.get(self.EIGEN3_URL, stream=True)
        with download_target_dir.open("wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir.name
    
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Pybind11Extension("pymlg.cpp._impl",
        ["pymlg/cpp/bindings.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs=[
            # Path to pybind11 headers
            get_eigen_include(),
            get_pybind_include(),
        ]
        ),
]

setup(
    name="pymlg",
    version=__version__,
    author="Charles C. Cossette",
    author_email="charles.cossette@mail.mcgill.ca",
    url="https://github.com/decargroup/pymlg",
    description="Lie group functions for python.",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test":["pytest", "jax", "jaxlib"]},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.1",
        ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    project_urls={
        "Source": "https://github.com/decargroup/pymlg",
        "Tracker": "https://github.com/decargroup/pymlg/issues",
        "Documentation": "https://decargroup.github.io/pymlg/"
    },
)