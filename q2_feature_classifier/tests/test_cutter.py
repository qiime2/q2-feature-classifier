# ----------------------------------------------------------------------------
# Copyright (c) 2016--, Ben Kaehler
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import types
import unittest
import os.path
from itertools import zip_longest

from qiime.plugins import feature_classifier
from qiime.sdk import Artifact
from q2_types import FeatureData, Sequence, PairedEndSequence


class CutterTests(unittest.TestCase):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    def test_extract_reads(self):
        # extract_reads should generate a FeatureData(Sequence) full of reads

        sequences = Artifact.load(os.path.join(self.data_dir,
                                               '85_ref_feat.qza'))
        read_length = 75
        f_primer = 'AGAGTTTGATCMTGGCTCAG'
        r_primer = 'GCTGCCTCCCGTAGGAGT'
        extract_reads = feature_classifier.methods.extract_reads
        reads = extract_reads(sequences, read_length, f_primer, r_primer)
        self.assertEqual(reads.type, FeatureData[Sequence])
        inseqs = sequences.view(types.GeneratorType)
        outseqs = reads.view(types.GeneratorType)
        for inseq, outseq in zip_longest(inseqs, outseqs):
            self.assertIsNotNone(inseq)
            self.assertIsNotNone(outseq)
            self.assertEqual(len(outseq), read_length)

    def test_paired_end_extract_reads(self):
        # extract_reads should generate a FeatureData(Sequence) full of reads

        sequences = Artifact.load(os.path.join(self.data_dir,
                                               '85_ref_feat.qza'))
        read_length = 75
        f_primer = 'AGAGTTTGATCMTGGCTCAG'
        r_primer = 'GCTGCCTCCCGTAGGAGT'
        extract_paired_end_reads = \
            feature_classifier.methods.extract_paired_end_reads
        reads = extract_paired_end_reads(sequences, read_length,
                                         f_primer, r_primer)
        self.assertEqual(reads.type, FeatureData[PairedEndSequence])
        inseqs = sequences.view(types.GeneratorType)
        outseqs = reads.view(types.GeneratorType)
        for inseq, outseq in zip_longest(inseqs, outseqs):
            self.assertIsNotNone(inseq)
            self.assertIsNotNone(outseq)
            self.assertEqual(len(outseq[0]), read_length)
            self.assertEqual(len(outseq[1]), read_length)
