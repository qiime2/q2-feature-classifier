import pkg_resources

from ._classifier import classify

__version__ = pkg_resources.require('q2_classifier')[0].version

__all__ = ['classify']
