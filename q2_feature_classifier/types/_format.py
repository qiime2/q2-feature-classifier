# ----------------------------------------------------------------------------
# Copyright (c) 2016-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import model
from ..plugin_setup import plugin, citations


class BLASTDBFileFmtV5(model.BinaryFileFormat):
    # We do not have a good way to validate the individual blastdb files.
    # TODO: could wire up `blastdbcheck` to do a deep check when level=max
    # but this must be done on the directory, not individual files.
    # For now validation be done at the DirFmt level on file extensions.
    def _validate_(self, level):
        pass


class BLASTDBDirFmtV5(model.DirectoryFormat):
    idx1 = model.File(r'.+\.ndb', format=BLASTDBFileFmtV5)
    idx2 = model.File(r'.+\.nhr', format=BLASTDBFileFmtV5)
    idx3 = model.File(r'.+\.nin', format=BLASTDBFileFmtV5)
    idx4 = model.File(r'.+\.not', format=BLASTDBFileFmtV5)
    idx5 = model.File(r'.+\.nsq', format=BLASTDBFileFmtV5)
    idx6 = model.File(r'.+\.ntf', format=BLASTDBFileFmtV5)
    idx7 = model.File(r'.+\.nto', format=BLASTDBFileFmtV5)


plugin.register_views(BLASTDBDirFmtV5,
                      citations=[citations['camacho2009blast+']])
