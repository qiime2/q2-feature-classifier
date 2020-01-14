# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import skbio

from qiime2.sdk import Artifact
from qiime2.plugins.feature_classifier.actions import extract_reads
from q2_types.feature_data._format import DNAFASTAFormat

from . import FeatureClassifierTestPluginBase


class CutterTests(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()
        self.sequences = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('dna-sequences.fasta'))

        self.mixed_sequences = Artifact.import_data(
                'FeatureData[Sequence]',
                self.get_data_path('dna-sequences-mixed.fasta'))

        self.f_primer = 'AGAGA'
        self.r_primer = 'GCTGC'

        self.amplicons = ['ACGT', 'AAGT', 'ACCT', 'ACGG', 'ACTT']

    def _test_results(self, results):
        for i, result in enumerate(
                skbio.io.read(str(results.reads.view(DNAFASTAFormat)),
                              format='fasta')):
            self.assertEqual(str(result), self.amplicons[i])

    def test_extract_reads_expected(self):
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=4)

        self._test_results(results)

    def test_extract_reads_expected_forward(self):
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=4, read_orientation='forward')

        self._test_results(results)

    def test_extract_mixed(self):
        results = extract_reads(
            self.mixed_sequences, f_primer=self.f_primer,
            r_primer=self.r_primer, min_length=4)

        self._test_results(results)

    def test_extract_reads_expected_reverse(self):
        reverse_sequences = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('dna-sequences-reverse.fasta'))

        results = extract_reads(
            reverse_sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=4, read_orientation='reverse')

        self._test_results(results)

    def test_extract_reads_manual_batch_size(self):
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=4, batch_size=10)

        self._test_results(results)

    def test_extract_reads_two_jobs(self):
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=4, n_jobs=2)

        self._test_results(results)

    def test_extract_reads_expected_degenerate_primers(self):
        degenerate_f_primer = 'WWWWW'
        degenerate_r_primer = 'SSSSS'

        degenerate_sequences = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('dna-sequences-degenerate-primers.fasta'))

        results = extract_reads(
            degenerate_sequences, f_primer=degenerate_f_primer,
            r_primer=degenerate_r_primer, min_length=4)

        self._test_results(results)

    def test_extract_reads_fail_identity(self):
        with self.assertRaisesRegex(RuntimeError, "No matches found"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                min_length=4, identity=1)

    def test_extract_reads_fail_min_length(self):
        with self.assertRaisesRegex(RuntimeError, "No matches found"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                min_length=5)

    def test_extract_reads_fail_max_length(self):
        with self.assertRaisesRegex(RuntimeError, "No matches found"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                max_length=1)

    def test_extract_reads_fail_trim_entire_read(self):
        with self.assertRaisesRegex(RuntimeError, "No matches found"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                trim_left=4)

    def test_extract_reads_fail_min_len_greater_than_trunc_len(self):
        with self.assertRaisesRegex(ValueError, "minimum length setting"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                trunc_len=1)
