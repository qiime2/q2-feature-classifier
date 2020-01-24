# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import Plugin, Citations

import q2_feature_classifier

citations = Citations.load('citations.bib', package='q2_feature_classifier')
plugin = Plugin(
    name='feature-classifier',
    version=q2_feature_classifier.__version__,
    website='https://github.com/qiime2/q2-feature-classifier',
    package='q2_feature_classifier',
    description=('This QIIME 2 plugin supports taxonomic '
                 'classification of features using a variety '
                 'of methods, including Naive Bayes, vsearch, '
                 'and BLAST+.'),
    short_description='Plugin for taxonomic classification.',
    citations=[citations['bokulich2018optimizing']]
)
