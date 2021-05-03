"""
spbla library state wrapper.
Provides global access points and initialization logic for library API.
"""

import os
import ctypes
import pathlib
from . import bridge

__all__ = [
    "singleton",
    "loaded_dll",
    "init_wrapper"
]

# Wrapper
singleton = None

# Exposed imported dll (owned by _lib_wrapper)
loaded_dll = None


def init_wrapper():
    global singleton
    global loaded_dll

    singleton = Wrapper()
    loaded_dll = singleton.loaded_dll


class Wrapper:
    """
    Instance of this class represents the main access point
    to the C spbla library. It stores:

     - Ctypes imported dll instance
     - Ctypes functions declarations
     - Import path

    At exit the library is properly finalized.
    Instance of the spbla library in released and the shared library is closed.
    """

    def __init__(self):
        self.loaded_dll = None
        self.backend = "default"

        try:
            # Try from config if present
            self.load_path = os.environ["SPBLA_PATH"]
        except KeyError:
            # Fallback to package directory
            source_path = pathlib.Path(__file__).resolve()
            self.load_path = Wrapper.__get_lib_path(source_path.parent)

        try:
            # Check, if user has backend-type preferences
            self.backend = str(os.environ["SPBLA_BACKEND"]).lower()
        except KeyError:
            pass

        assert self.load_path
        assert self.backend

        self.loaded_dll = bridge.load_and_configure(self.load_path)
        self.__setup_library()

    def __del__(self):
        self.__release_library()

    def __setup_library(self):
        status = self.loaded_dll.spbla_Initialize(ctypes.c_uint(bridge.get_init_hints(self.backend)))
        bridge.check(status)

    def __release_library(self):
        status = self.loaded_dll.spbla_Finalize()
        bridge.check(status)

    @classmethod
    def __get_lib_path(cls, prefix):
        for entry in os.listdir(prefix):
            if "spbla" in str(entry):
                return prefix / entry

        return None
