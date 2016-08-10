# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


import importlib
import pkg_resources

__version__ = pkg_resources.require('q2_feature_classifier')[0].version

importlib.import_module('q2_feature_classifier._classifier')
importlib.import_module('q2_feature_classifier._cutter')

__all__ = ['classify', 'extract_reads']
