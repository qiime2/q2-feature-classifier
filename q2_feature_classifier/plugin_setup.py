# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Plugin

import q2_feature_classifier


plugin = Plugin(
    name='feature-classifier',
    version=q2_feature_classifier.__version__,
    website='https://github.com/qiime2/q2-feature-classifier',
    package='q2_feature_classifier'
)
