# ----------------------------------------------------------------------------
# Copyright (c) 2016-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
from itertools import product, islice

import skbio
from qiime2.plugins import feature_classifier
from qiime2.sdk import Artifact
from q2_types.feature_data import FeatureData, Sequence, DNAIterator

from . import FeatureClassifierTestPluginBase


class CutterTests(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()
        seqs = skbio.io.read(self.get_data_path('dna-sequences.fasta'),
                             format='fasta', constructor=skbio.DNA)
        tmpseqs = os.path.join(self.temp_dir.name, 'temp-seqs.fasta')
        skbio.io.write((s for s in islice(seqs, 10)), 'fasta', tmpseqs)
        self.sequences = Artifact.import_data('FeatureData[Sequence]', tmpseqs)

    def test_extract_reads(self):
        # extract_reads should generate a FeatureData(Sequence) full of reads
        f_primer = 'AGAGTTTGATCMTGGCTCAG'
        r_primer = 'GCTGCCTCCCGTAGGAGT'
        extract_reads = feature_classifier.methods.extract_reads
        inseqs = list(self.sequences.view(DNAIterator))

        trunc_lens = 0, 75, 5
        trim_lefts = 0, 5
        raw_lens = []
        for trunc_len, trim_left in product(trunc_lens, trim_lefts):
            if trunc_len == trim_left and trunc_len > 0:
                with self.assertRaisesRegex(RuntimeError, "No matches found"):
                    result = extract_reads(
                        self.sequences, f_primer=f_primer, r_primer=r_primer,
                        trunc_len=trunc_len, trim_left=trim_left, identity=0.9,
                        min_length=0, max_length=0)
                continue
            result = extract_reads(
                self.sequences, f_primer=f_primer, r_primer=r_primer,
                trunc_len=trunc_len, trim_left=trim_left, identity=0.9,
                min_length=0, max_length=0)
            self.assertEqual(result.reads.type, FeatureData[Sequence])
            outseqs = list(result.reads.view(DNAIterator))
            self.assertGreater(len(outseqs), 0)
            self.assertLessEqual(len(outseqs), len(inseqs))
            self.assertTrue(bool(outseqs))
            if trunc_len != 0:
                for seq in outseqs:
                    self.assertEqual(len(seq), trunc_len - trim_left)
            elif trim_left == 0:
                for seq in outseqs:
                    raw_lens.append(len(seq))
            else:
                for seq, raw_len in zip(outseqs, raw_lens):
                    self.assertEqual(len(seq), raw_len - trim_left)
            # test that length filtering is working
            with self.assertRaisesRegex(RuntimeError, "No matches found"):
                result = extract_reads(
                    self.sequences, f_primer=f_primer, r_primer=r_primer,
                    trunc_len=trunc_len, trim_left=trim_left, identity=0.9,
                    min_length=500, max_length=0)
            with self.assertRaisesRegex(RuntimeError, "No matches found"):
                result = extract_reads(
                    self.sequences, f_primer=f_primer, r_primer=r_primer,
                    trunc_len=trunc_len, trim_left=trim_left, identity=0.9,
                    min_length=0, max_length=20)
