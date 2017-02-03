# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugins import feature_classifier
from qiime2.sdk import Artifact
from q2_types.feature_data import FeatureData, Sequence, DNAIterator

from . import FeatureClassifierTestPluginBase


class CutterTests(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()
        self.sequences = Artifact.import_data(
            'FeatureData[Sequence]', self.get_data_path('dna-sequences.fasta'))

    def test_extract_reads(self):
        # extract_reads should generate a FeatureData(Sequence) full of reads
        length = 75
        f_primer = 'AGAGTTTGATCMTGGCTCAG'
        r_primer = 'GCTGCCTCCCGTAGGAGT'
        extract_reads = feature_classifier.methods.extract_reads
        result = extract_reads(self.sequences, f_primer=f_primer,
                               r_primer=r_primer, length=length, identity=0.9)
        self.assertEqual(result.reads.type, FeatureData[Sequence])
        inseqs = list(self.sequences.view(DNAIterator))
        outseqs = list(result.reads.view(DNAIterator))
        self.assertLessEqual(len(outseqs), len(inseqs))
        self.assertTrue(bool(outseqs))
        for seq in outseqs:
            self.assertEqual(len(seq), length)
