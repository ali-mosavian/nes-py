"""The nes-py NES emulator for Python 3."""
from nes_py.nes_env import NESEnv

# explicitly define the outward facing API of this package
__all__ = [NESEnv.__name__]