# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages
import versioneer

setup(
    name="q2-feature-classifier",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    install_requires=['qiime2 == 2017.3.*', 'q2-types == 2017.3.*',
                      'scikit-bio', 'biom-format >= 2.1.5, < 2.2.0',
                      'scikit-learn'],
    author="Ben Kaehler",
    author_email="kaehler@gmail.com",
    description="Functionality for taxonomic classification",
    license='BSD-3-Clause',
    url="https://qiime2.org",
    entry_points={
        'qiime2.plugins':
        ['q2-feature-classifier=q2_feature_classifier.plugin_setup:plugin']
    },
    zip_safe=False,
)
