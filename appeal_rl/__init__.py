import sys
sys.path.append('..')
sys.path.append('../..')
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
