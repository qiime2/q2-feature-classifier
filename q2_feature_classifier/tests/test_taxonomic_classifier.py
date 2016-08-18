# ----------------------------------------------------------------------------
# Copyright (c) 2016--, QIIME development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os.path
import tempfile
import unittest

from qiime.sdk import Artifact, PluginManager


class TypesTests(unittest.TestCase):

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data')

    def test_load_save(self):
        # confirm that
        # the artifact can be loaded and a new artifact saved and reloaded
        # load example artifact
        artifact_fp = os.path.join(self.data_dir,
                                   'taxonomic-classifier.qza')
        a = Artifact.load(artifact_fp)
        with tempfile.NamedTemporaryFile(suffix='.qza') as f:
            # save loaded artifact
            a.save(f.name)
            # reload saved artifact
            Artifact.load(f.name)

    def test_data_layout_readers_and_writers(self):
        # Confirm that for the example artifact,
        # all registered views of that artifact can be loaded without error
        # (the data in the view is not yet tested, just that it loads without
        # error). Also test that for each registered writer, an appropriate
        # view can be written. Note: the writer tests assume there is a
        # corresponding reader for the writer's type, which may not always be
        # true (but is for the types currently in this repo). This test may
        # need to be revised if this assumption doesn't hold in the future.
        pm = PluginManager()
        # load example artifact
        artifact_fp = os.path.join(self.data_dir,
                                   'taxonomic-classifier.qza')
        a = Artifact.load(artifact_fp)
        data_layout = pm.get_data_layout(a.type)
        for view_type in data_layout.readers:
            view = a.view(view_type)
            self.assertIs(type(view), view_type)
        for view_type in data_layout.writers:
            view = a.view(view_type)
            self.assertIs(type(view), view_type)
            Artifact._from_view(view, a.type, None)


if __name__ == "__main__":
    unittest.main()
