try:
    import sys

    if sys.version_info >= (3, 8):
        from importlib import metadata
    else:
        import importlib_metadata as metadata
    __version__ = metadata.version('qcdevol')
except:
    # less precise default if fail
    __version__ = '>=4.0'

from .qcdevol import *

