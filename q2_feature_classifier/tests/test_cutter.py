# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from itertools import zip_longest

from qiime2.plugins import feature_classifier
from qiime2.sdk import Artifact
from q2_types.feature_data import (
    FeatureData, Sequence, PairedEndSequence, DNAIterator, PairedDNAIterator)


from . import FeatureClassifierTestPluginBase


class CutterTests(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()
        self.sequences = Artifact.import_data(
            'FeatureData[Sequence]', self.get_data_path('dna-sequences.fasta'))

    def test_extract_reads(self):
        # extract_reads should generate a FeatureData(Sequence) full of reads
        read_length = 75
        f_primer = 'AGAGTTTGATCMTGGCTCAG'
        r_primer = 'GCTGCCTCCCGTAGGAGT'
        extract_reads = feature_classifier.methods.extract_reads
        result = extract_reads(self.sequences, read_length, f_primer, r_primer)
        self.assertEqual(result.reads.type, FeatureData[Sequence])
        inseqs = self.sequences.view(DNAIterator)
        outseqs = result.reads.view(DNAIterator)
        for inseq, outseq in zip_longest(inseqs, outseqs):
            self.assertIsNotNone(inseq)
            self.assertIsNotNone(outseq)
            self.assertEqual(len(outseq), read_length)

    def test_paired_end_extract_reads(self):
        # extract_reads should generate a FeatureData(Sequence) full of reads
        read_length = 75
        f_primer = 'AGAGTTTGATCMTGGCTCAG'
        r_primer = 'GCTGCCTCCCGTAGGAGT'
        extract_paired_end_reads = \
            feature_classifier.methods.extract_paired_end_reads
        results = extract_paired_end_reads(self.sequences, read_length,
                                           f_primer, r_primer)
        self.assertEqual(results.reads.type, FeatureData[PairedEndSequence])
        inseqs = self.sequences.view(DNAIterator)
        outseqs = results.reads.view(PairedDNAIterator)
        for inseq, outseq in zip_longest(inseqs, outseqs):
            self.assertIsNotNone(inseq)
            self.assertIsNotNone(outseq)
            self.assertEqual(len(outseq[0]), read_length)
            self.assertEqual(len(outseq[1]), read_length)
