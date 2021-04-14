"""
Entry point for `pyspbla` package.
pyspbla is python wrapper for spbla library.

The backend spbla compiled library is initialized automatically
on package import into runtime. State of the library is managed
by the wrapper class. All resources are unloaded automatically on
python runtime exit.

Exported primitives:
- matrix (sparse matrix of boolean values)

For more information refer to:
- spbla project: https://github.com/JetBrains-Research/spbla
- bug tracker: https://github.com/JetBrains-Research/spbla/issues

License:
- MIT
"""

from .wrapper import *
from .utils import *
from .matrix import *
from .io import *
from .gviz import *

# Setup global module state
init_wrapper()
