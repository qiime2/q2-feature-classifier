# ----------------------------------------------------------------------------
# Copyright (c) 2016-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import SemanticType
from . import BLASTDBDirFmtV5
from ..plugin_setup import plugin


BLASTDB = SemanticType('BLASTDB')

plugin.register_semantic_types(BLASTDB)
plugin.register_artifact_class(BLASTDB, BLASTDBDirFmtV5)
