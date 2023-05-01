# ----------------------------------------------------------------------------
# Copyright (c) 2016-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
import itertools
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

    # borrowed from q2-types
    def get_basename(self):
        paths = [str(x.relative_to(self.path)) for x in self.path.iterdir()]
        prefix = os.path.splitext(_get_prefix(paths))[0]
        return prefix


# SO: https://stackoverflow.com/a/6718380/579416
def _get_prefix(strings):
    def all_same(x):
        return all(x[0] == y for y in x)

    char_tuples = zip(*strings)
    prefix_tuples = itertools.takewhile(all_same, char_tuples)
    return ''.join(x[0] for x in prefix_tuples)


plugin.register_views(BLASTDBDirFmtV5,
                      citations=[citations['camacho2009blast+']])
