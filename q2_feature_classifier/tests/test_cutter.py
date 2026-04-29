# ----------------------------------------------------------------------------
# Copyright (c) 2016-2026, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import skbio
import qiime2

from qiime2.sdk import Artifact
from qiime2.plugins.feature_classifier.actions import extract_reads
from q2_types.feature_data import DNAFASTAFormat

from q2_feature_classifier._cutter import (
    _create_asymmetric_primer_substitution_matrix,
    _match_percent,
    _align_primer,
    _approx_match,
)
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

    def test_extract_reads_expected_trim_right(self):
        """Tests expected behavior of trim_right option"""
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=3, trim_right=1)

        for i, result in enumerate(
                skbio.io.read(str(results.reads.view(DNAFASTAFormat)),
                              format='fasta')):
            self.assertEqual(str(result), self.amplicons[i][:-1])

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

    def test_extract_reads_fail_trim_left_entire_read(self):
        with self.assertRaisesRegex(RuntimeError, "No matches found"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                trim_left=4)

    def test_extract_reads_fail_trim_right_entire_read(self):
        with self.assertRaisesRegex(RuntimeError, "No matches found"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                trim_right=4)

    def test_extract_reads_fail_trim_both_entire_read(self):
        with self.assertRaisesRegex(RuntimeError, "No matches found"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                trim_left=2, trim_right=2)

    def test_extract_reads_fail_min_len_greater_than_trunc_len(self):
        with self.assertRaisesRegex(ValueError, "minimum length setting"):
            extract_reads(
                self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
                trunc_len=1)

    def test_extract_reads_stats_returned(self):
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=4)
        self.assertIsNotNone(results.read_extraction_stats)

    def test_extract_reads_stats_schema(self):
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=4)
        df = results.read_extraction_stats.view(qiime2.Metadata).to_dataframe()
        expected_columns = {
            'outcome', 'match-orientation', 'match-method',
            'f-primer-start', 'f-primer-end', 'r-primer-start', 'r-primer-end',
            'f-primer-match-pct', 'r-primer-match-pct',
            'amplicon-length-pre-trim', 'amplicon-length-post-trim',
            'input-sequence-length',
        }
        self.assertEqual(set(df.columns), expected_columns)
        self.assertEqual(len(df), 5)

    def test_extract_reads_stats_all_extracted(self):
        # dna-sequences.fasta has exact f_primer but an approximate r_primer
        # site (GCAGG vs RC(GCTGC)=GCAGC), so match-method is 'approximate'.
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=4)
        df = results.read_extraction_stats.view(qiime2.Metadata).to_dataframe()
        self.assertTrue((df['outcome'] == 'extracted').all())
        self.assertTrue((df['match-method'] == 'approximate').all())
        self.assertTrue((df['match-orientation'] == 'forward').all())
        self.assertTrue((df['f-primer-match-pct'] == 1.0).all())
        self.assertTrue((df['r-primer-match-pct'] > 0.7).all())
        self.assertTrue((df['amplicon-length-pre-trim'] == 4).all())
        self.assertTrue((df['amplicon-length-post-trim'] == 4).all())
        # primer position columns are populated for all extracted sequences
        self.assertFalse(df['f-primer-start'].isna().any())
        self.assertFalse(df['r-primer-end'].isna().any())
        self.assertTrue(False, " TODO: Make this test more specific!")

    def test_extract_reads_stats_orientation(self):
        results = extract_reads(
            self.mixed_sequences, f_primer=self.f_primer,
            r_primer=self.r_primer, min_length=4)
        df = results.read_extraction_stats.view(qiime2.Metadata).to_dataframe()
        orientations = set(df['match-orientation'].unique())
        self.assertIn('forward', orientations)
        self.assertIn('reverse', orientations)

    def test_extract_reads_stats_trim_reduces_post_trim_length(self):
        results = extract_reads(
            self.sequences, f_primer=self.f_primer, r_primer=self.r_primer,
            min_length=3, trim_right=1)
        df = results.read_extraction_stats.view(qiime2.Metadata).to_dataframe()
        self.assertTrue((df['amplicon-length-pre-trim'] == 4).all())
        self.assertTrue((df['amplicon-length-post-trim'] == 3).all())

    def test_extract_reads_stats_excluded_max_length(self):
        # Use max_length=3 so amplicons of length 4 are excluded, but leave
        # min_length=0 so a few bases could pass — here all are excluded.
        # We have to catch the RuntimeError and inspect stats separately via
        # the lower-level callable since the QIIME 2 action raises before
        # returning artifacts when no reads are extracted.
        from q2_feature_classifier._cutter import _gen_reads
        seq = skbio.DNA('AGAGAACGTGCTGC', metadata={'id': 'test-seq'})
        amp, stats = _gen_reads(seq, self.f_primer, self.r_primer,
                                trim_right=0, trunc_len=0, trim_left=0,
                                identity=0.7, min_length=0, max_length=3,
                                read_orientation='forward')
        self.assertIsNone(amp)
        self.assertEqual(stats['outcome'], 'excluded-max-length')
        self.assertEqual(stats['amplicon-length-pre-trim'], 4)
        self.assertIsNone(stats['amplicon-length-post-trim'])

    def test_extract_reads_stats_no_primer_match(self):
        from q2_feature_classifier._cutter import _gen_reads
        seq = skbio.DNA('TTTTTTTTTTTTTT', metadata={'id': 'test-seq'})
        amp, stats = _gen_reads(seq, self.f_primer, self.r_primer,
                                trim_right=0, trunc_len=0, trim_left=0,
                                identity=0.9, min_length=0, max_length=0,
                                read_orientation='forward')
        self.assertIsNone(amp)
        self.assertEqual(stats['outcome'], 'no-primer-match')
        self.assertEqual(stats['match-orientation'], 'none')
        self.assertIsNone(stats['f-primer-start'])

    def _assert_gen_reads_rc_symmetry(self, seq_str, expected_method):
        from q2_feature_classifier._cutter import _gen_reads
        seq = skbio.DNA(seq_str, metadata={'id': 'test-seq'})
        rc_seq = seq.reverse_complement()

        amp_fwd, stats_fwd = _gen_reads(
            seq, self.f_primer, self.r_primer,
            trim_right=0, trunc_len=0, trim_left=0,
            identity=0.7, min_length=0, max_length=0,
            read_orientation='both')
        amp_rev, stats_rev = _gen_reads(
            rc_seq, self.f_primer, self.r_primer,
            trim_right=0, trunc_len=0, trim_left=0,
            identity=0.7, min_length=0, max_length=0,
            read_orientation='both')

        self.assertIsNotNone(amp_fwd)
        self.assertIsNotNone(amp_rev)
        self.assertEqual(str(amp_fwd), str(amp_rev))
        self.assertEqual(stats_fwd['match-method'], expected_method)
        self.assertEqual(stats_rev['match-method'], expected_method)
        self.assertEqual(stats_fwd['match-orientation'], 'forward')
        self.assertEqual(stats_rev['match-orientation'], 'reverse')

    def test_gen_reads_rc_symmetry_exact(self):
        # Trailing 'GCAGC' = RC('GCTGC') is an exact reverse-primer site on
        # the forward strand, so the exact-match path is exercised.
        self._assert_gen_reads_rc_symmetry('AGAGAACGTGCAGC', 'exact')

    def test_gen_reads_rc_symmetry_approximate(self):
        # Trailing 'GCTGC' differs from RC('GCTGC')='GCAGC' at one position,
        # forcing the approximate-match path (4/5 = 0.8 >= identity=0.7).
        self._assert_gen_reads_rc_symmetry('AGAGAACGTGCTGC', 'approximate')


class TestCreateAsymmetricPrimerSubstitutionMatrix(
        FeatureClassifierTestPluginBase
):
    @classmethod
    def setUpClass(cls):
        cls.chars = sorted(skbio.DNA.definite_chars) + \
            sorted(skbio.DNA.degenerate_chars)

    def test_returns_substitution_matrix(self):
        sm = _create_asymmetric_primer_substitution_matrix()
        self.assertIsInstance(sm, skbio.SubstitutionMatrix)

    def test_custom_match_mismatch_values_produce_different_matrix(self):
        sm_default = _create_asymmetric_primer_substitution_matrix()
        sm_custom = _create_asymmetric_primer_substitution_matrix(
            match=5, mismatch=-1)
        self.assertEqual({2, -3}, set(np.unique(sm_default.scores)))
        self.assertEqual({5, -1}, set(np.unique(sm_custom.scores)))

    def test_matrix_covers_all_dna_chars(self):
        sm = _create_asymmetric_primer_substitution_matrix()
        expected_n = (len(skbio.DNA.definite_chars)
                      + len(skbio.DNA.degenerate_chars))
        self.assertEqual(sm.scores.shape, (expected_n, expected_n))

    def test_degenerate_primer_matches_represented_definite_target(self):
        # W represents {A, T}; should score 'match' when target is A or T
        # and 'mismatch' when target is C or G
        sm = _create_asymmetric_primer_substitution_matrix(
            match=2, mismatch=-3)
        idx = {c: i for i, c in enumerate(self.chars)}
        self.assertEqual(sm.scores[idx['W'], idx['A']], 2)
        self.assertEqual(sm.scores[idx['W'], idx['T']], 2)
        self.assertEqual(sm.scores[idx['W'], idx['C']], -3)
        self.assertEqual(sm.scores[idx['W'], idx['G']], -3)

    def test_definite_primer_vs_degenerate_target_is_always_mismatch(self):
        # A degenerate character in the target is always a mismatch regardless
        # of which base the primer has
        sm = _create_asymmetric_primer_substitution_matrix(
            match=2, mismatch=-3)
        idx = {c: i for i, c in enumerate(self.chars)}
        self.assertEqual(sm.scores[idx['A'], idx['W']], -3)
        self.assertEqual(sm.scores[idx['A'], idx['N']], -3)
        self.assertEqual(sm.scores[idx['W'], idx['N']], -3)

    def test_definite_primer_vs_definite_target_match_and_mismatch(self):
        sm = _create_asymmetric_primer_substitution_matrix(
            match=2, mismatch=-3)
        idx = {c: i for i, c in enumerate(self.chars)}
        self.assertEqual(sm.scores[idx['A'], idx['A']], 2)
        self.assertEqual(sm.scores[idx['A'], idx['C']], -3)


class TestMatchPercent(FeatureClassifierTestPluginBase):

    def test_perfect_match(self):
        result = _match_percent(skbio.DNA('ACGT'), skbio.DNA('ACGT'))
        self.assertEqual(result, 1.0)

    def test_no_match(self):
        result = _match_percent(skbio.DNA('AAAA'), skbio.DNA('CCCC'))
        self.assertEqual(result, 0.0)

    def test_partial_match(self):
        # 3 of 4 positions match
        result = _match_percent(skbio.DNA('ACGT'), skbio.DNA('ACGG'))
        self.assertAlmostEqual(result, 0.75)

    def test_degenerate_primer_char_matches_represented_target(self):
        # W = {A, T}; counts as a match when target is A or T
        self.assertEqual(_match_percent(skbio.DNA('W'), skbio.DNA('A')), 1.0)
        self.assertEqual(_match_percent(skbio.DNA('W'), skbio.DNA('T')), 1.0)

    def test_degenerate_primer_char_no_match(self):
        # W = {A, T}; does not match C or G
        self.assertEqual(_match_percent(skbio.DNA('W'), skbio.DNA('C')), 0.0)
        self.assertEqual(_match_percent(skbio.DNA('W'), skbio.DNA('G')), 0.0)

    def test_n_in_primer_matches_any_definite_target(self):
        # N represents all bases
        for base in 'ACGT':
            self.assertEqual(
                _match_percent(skbio.DNA('N'), skbio.DNA(base)), 1.0)

    def test_degenerate_target_char_skipped_reduces_score(self):
        # Degenerate char in TARGET is skipped (not counted as a match),
        # but the denominator is still len(primer), reducing the score.
        # ACGT vs ACNT: N at position 2 is skipped → 3 matches out of 4
        result = _match_percent(skbio.DNA('ACGT'), skbio.DNA('ACNT'))
        self.assertAlmostEqual(result, 0.75)


class TestAlignPrimer(FeatureClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        self.sm = _create_asymmetric_primer_substitution_matrix()

    def test_perfect_forward_match_percent(self):
        primer = skbio.DNA('AAAA')
        target = skbio.DNA('GGGGAAAAGGGG')
        result = _align_primer(primer, target, self.sm, reverse=False)
        self.assertAlmostEqual(result.match_percent, 1.0)

    def test_perfect_reverse_match_percent(self):
        # rc('TTTT') = AAAA, which matches the AAAA region perfectly
        primer = skbio.DNA('TTTT')
        target = skbio.DNA('GGGGAAAAGGGG')
        result = _align_primer(primer, target, self.sm, reverse=True)
        self.assertAlmostEqual(result.match_percent, 1.0)

    def test_partial_forward_match_percent(self):
        # AAAC aligns to AAAA in target — 3 of 4 positions match
        primer = skbio.DNA('AAAC')
        target = skbio.DNA('GGGGAAAAGGGG')
        result = _align_primer(primer, target, self.sm, reverse=False)
        self.assertAlmostEqual(result.match_percent, 0.75)

    def test_forward_amplicon_pos_is_after_primer(self):
        # Forward mode returns a position such that target[pos:] is the content
        # after the primer
        primer = skbio.DNA('AAAA')
        target = skbio.DNA('AAAAGGGG')
        result = _align_primer(primer, target, self.sm, reverse=False)
        self.assertEqual(str(target[result.amplicon_pos:]), 'GGGG')

    def test_reverse_amplicon_pos_is_before_primer(self):
        # Reverse mode: rc('GGGG') = CCCC; CCCC matches the end of target
        # The returned position should be such that target[:pos] is the
        # amplicon
        primer = skbio.DNA('GGGG')
        target = skbio.DNA('AAAACCCC')
        result = _align_primer(primer, target, self.sm, reverse=True)
        self.assertEqual(str(target[:result.amplicon_pos]), 'AAAA')

    def test_forward_and_reverse_return_different_positions(self):
        primer = skbio.DNA('AAAA')
        target = skbio.DNA('CCCCAAAAGGGG')
        fwd = _align_primer(primer, target, self.sm, reverse=False)
        rev = _align_primer(primer, target, self.sm, reverse=True)
        self.assertNotEqual(fwd.amplicon_pos, rev.amplicon_pos)

    def test_forward_primer_start_and_end_in_target(self):
        # AAAA aligns at positions 4-8 in CCCCAAAAGGGG
        primer = skbio.DNA('AAAA')
        target = skbio.DNA('CCCCAAAAGGGG')
        result = _align_primer(primer, target, self.sm, reverse=False)
        self.assertEqual(str(target[result.primer_start:result.primer_end]),
                         'AAAA')
        self.assertTrue(False, "TODO: add diversity to this primer seq")

    def test_reverse_primer_start_and_end_in_target(self):
        # rc('GGGG') = CCCC matches the last 4 bases of AAAACCCC
        primer = skbio.DNA('GGGG')
        target = skbio.DNA('AAAACCCC')
        result = _align_primer(primer, target, self.sm, reverse=True)
        self.assertEqual(str(target[result.primer_start:result.primer_end]),
                         'CCCC')
        self.assertTrue(False, "TODO: add diversity to this primer seq")


class TestApproxMatch(FeatureClassifierTestPluginBase):

    def test_both_primers_match_returns_correct_amplicon(self):
        # f_primer AAAA at start; rc(r_primer) = CCCC at end
        # expected amplicon = GGGG
        seq = skbio.DNA('AAAAGGGGCCCC')
        result = _approx_match(seq, skbio.DNA('AAAA'),
                               skbio.DNA('GGGG'), identity=0.9)
        self.assertIsNotNone(result)
        self.assertEqual(str(result[0]), 'GGGG')

    def test_both_primers_match_returns_stats(self):
        seq = skbio.DNA('AAAAGGGGCCCC')
        result = _approx_match(seq, skbio.DNA('AAAA'),
                               skbio.DNA('GGGG'), identity=0.9)
        self.assertIsNotNone(result)
        amp, f_start, f_end, r_start, r_end, f_pct, r_pct = result
        self.assertAlmostEqual(f_pct, 1.0)
        self.assertAlmostEqual(r_pct, 1.0)
        # forward primer spans first 4 bases
        self.assertEqual(str(seq[f_start:f_end]), 'AAAA')
        # reverse primer (rc CCCC) spans last 4 bases
        self.assertEqual(str(seq[r_start:r_end]), 'CCCC')

    def test_f_primer_below_identity_returns_none(self):
        # TTAA does not match the AAAA region of seq at 0.7 identity
        # (has 0.5 identity)
        seq = skbio.DNA('AAAAGGGGCCCC')
        result = _approx_match(seq, skbio.DNA('TTAA'),
                               skbio.DNA('GGGG'), identity=0.7)
        self.assertIsNone(result)

    def test_r_primer_below_identity_returns_none(self):
        # rc('AAAA') = TTTT, which does not match the CCCC end → 0.0
        seq = skbio.DNA('AAAAGGGGCCCC')
        result = _approx_match(seq, skbio.DNA('AAAA'),
                               skbio.DNA('AAAA'), identity=0.7)
        self.assertIsNone(result)

    def test_identity_checked_per_primer_not_combined(self):
        # f_primer AAAA matches AAAA perfectly (1.0).
        # r_primer TGGT → rc = ACCA; best match against CCCC end is 2/4 = 0.5.
        # Per-primer check at identity=0.7: 0.5 < 0.7 → None.
        # Under a combined-identity scheme: (1.0 + 0.5) / 2 = 0.75 ≥ 0.7
        # would have passed, so this test distinguishes the two behaviours.
        seq = skbio.DNA('AAAAGGGGCCCC')
        result = _approx_match(seq, skbio.DNA('AAAA'),
                               skbio.DNA('TGGT'), identity=0.7)
        self.assertIsNone(result)

    def test_identity_one_requires_perfect_match(self):
        seq = skbio.DNA('AAAAGGGGCCCC')
        self.assertIsNotNone(
            _approx_match(seq, skbio.DNA('AAAA'), skbio.DNA('GGGG'),
                          identity=1.0))
        # One mismatch in f_primer → should fail at identity=1.0
        self.assertIsNone(
            _approx_match(seq, skbio.DNA('AAAC'), skbio.DNA('GGGG'),
                          identity=1.0))
