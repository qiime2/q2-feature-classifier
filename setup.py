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
    version='0.0.0-dev',
    packages=find_packages(),
    install_requires=['biom-format >= 2.1.5, < 2.2.0',
                      'qiime >= 2.0.0', 'q2-types', 'scikit-bio',
                      'scikit-learn',],
    author="Ben Kaehler",
    author_email="kaehler@gmail.com",
    description="Functionality for taxonomic classification",
    license='BSD-3-Clause',
    url="https://github.com/BenKaehler/q2-feature-classifier",
    entry_points={
        'qiime.plugins':
        ['q2-feature-classifier=q2_feature_classifier.plugin_setup:plugin']
    }
)
