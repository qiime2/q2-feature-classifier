# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages

setup(
    name="q2-feature-classifier",
    version='0.0.7.dev0',
    packages=find_packages(),
    install_requires=['qiime >= 2.0.6', 'q2-types >= 0.0.6', 'scikit-bio',
                      'biom-format >= 2.1.5, < 2.2.0', 'scikit-learn'],
    author="Ben Kaehler",
    author_email="kaehler@gmail.com",
    description="Functionality for taxonomic classification",
    license='BSD-3-Clause',
    url="https://github.com/qiime2/q2-feature-classifier",
    entry_points={
        'qiime.plugins':
        ['q2-feature-classifier=q2_feature_classifier.plugin_setup:plugin']
    }
)
