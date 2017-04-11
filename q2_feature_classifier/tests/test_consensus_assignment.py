# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
from q2_feature_classifier._blast import classify_consensus_blast
from q2_feature_classifier._vsearch import classify_consensus_vsearch
from q2_feature_classifier._consensus_assignment import (
    _compute_consensus_annotation,
    _compute_consensus_annotations,
    _import_blast_format_assignments,
    _output_no_hits)
from q2_types.feature_data import DNAFASTAFormat
from . import FeatureClassifierTestPluginBase


class ConsensusAssignmentsTests(FeatureClassifierTestPluginBase):
    package = 'q2_feature_classifier.tests'

    def setUp(self):
        super().setUp()
        self.taxonomy_fp = self.get_data_path('taxonomy.tsv')
        self.taxonomy = pd.Series.from_csv(self.taxonomy_fp, sep='\t')
        self.reads_fp = self.get_data_path('se-dna-sequences.fasta')
        self.reads = DNAFASTAFormat(self.reads_fp, mode='r')

    # Make sure blast and vsearch produce expected outputs
    # but there is no "right" taxonomy assignment.
    def test_blast(self):
        result = classify_consensus_blast(self.reads, self.reads,
                                          self.taxonomy)
        res = result.Taxon.to_dict()
        tax = self.taxonomy.to_dict()
        right = 0.
        for taxon in res:
            right += tax[int(taxon)].startswith(res[taxon])
        self.assertGreater(right/len(res), 0.5)

    def test_vsearch(self):
        result = classify_consensus_vsearch(self.reads, self.reads,
                                            self.taxonomy)
        res = result.Taxon.to_dict()
        tax = self.taxonomy.to_dict()
        right = 0.
        for taxon in res:
            right += tax[int(taxon)].startswith(res[taxon])
        self.assertGreater(right/len(res), 0.5)


class ImportBlastAssignmentTests(FeatureClassifierTestPluginBase):

    def test_import_blast_format_assignments(self):
        in_ = ['# This is a blast comment line',
               's1\t111\t100.000\t1428\t0\t0\t1\t1428\t1\t1428\t0.0\t2638',
               's1\t112\t100.000\t1479\t0\t0\t1\t1479\t1\t1479\t0.0\t2732',
               's2\t113\t100.000\t1336\t0\t0\t1\t1336\t1\t1336\t0.0\t2468',
               's2\t114\t100.000\t1402\t0\t0\t1\t1402\t1\t1402\t0.0\t2582']
        ref = {'111': 'Aa;Bb;Cc',
               '112': 'Aa;Bb;Cc',
               '113': 'Aa;Dd;Ee',
               '114': 'Aa;Dd;Ff'}
        ref = pd.Series(ref)
        obs = _import_blast_format_assignments(in_, ref)
        exp = {'s1': [['Aa', 'Bb', 'Cc'], ['Aa', 'Bb', 'Cc']],
               's2': [['Aa', 'Dd', 'Ee'], ['Aa', 'Dd', 'Ff']]}
        self.assertEqual(obs, exp)


# This code has been ported from QIIME 1.9.1 with permission from @gregcaporaso
class ConsensusAnnotationTests(FeatureClassifierTestPluginBase):

    def test_varied_min_fraction(self):
        in_ = [['Ab', 'Bc', 'De'],
               ['Ab', 'Bc', 'Fg', 'Hi'],
               ['Ab', 'Bc', 'Fg', 'Jk']]

        actual = _compute_consensus_annotation(in_, 0.51, "Unassigned")
        expected = (['Ab', 'Bc', 'Fg'], 2. / 3.)
        self.assertEqual(actual, expected)

        # increased min_consensus_fraction yields decreased specificity
        in_ = [['Ab', 'Bc', 'De'],
               ['Ab', 'Bc', 'Fg', 'Hi'],
               ['Ab', 'Bc', 'Fg', 'Jk']]

        actual = _compute_consensus_annotation(in_, 0.99, "Unassigned")
        expected = (['Ab', 'Bc'], 1.0)
        self.assertEqual(actual, expected)

    def test_single_annotation(self):
        in_ = [['Ab', 'Bc', 'De']]

        actual = _compute_consensus_annotation(in_, 1.0, "Unassigned")
        expected = (['Ab', 'Bc', 'De'], 1.0)
        self.assertEqual(actual, expected)

        actual = _compute_consensus_annotation(in_, 0.501, "Unassigned")
        expected = (['Ab', 'Bc', 'De'], 1.0)
        self.assertEqual(actual, expected)

    def test_no_consensus(self):
        in_ = [['Ab', 'Bc', 'De'],
               ['Cd', 'Bc', 'Fg', 'Hi'],
               ['Ef', 'Bc', 'Fg', 'Jk']]

        actual = _compute_consensus_annotation(in_, 0.51, "Unassigned")
        expected = (['Unassigned'], 1.)
        self.assertEqual(actual, expected)

        actual = _compute_consensus_annotation(
                    in_, 0.51, unassignable_label="Hello world!")
        expected = (['Hello world!'], 1.)
        self.assertEqual(actual, expected)

    def test_invalid_min_consensus_fraction(self):
        in_ = [['Ab', 'Bc', 'De'],
               ['Ab', 'Bc', 'Fg', 'Hi'],
               ['Ab', 'Bc', 'Fg', 'Jk']]
        self.assertRaises(ValueError, _compute_consensus_annotation, in_,
                          0.50, "Unassigned")
        self.assertRaises(ValueError, _compute_consensus_annotation, in_,
                          0.00, "Unassigned")
        self.assertRaises(ValueError, _compute_consensus_annotation, in_,
                          -0.1, "Unassigned")

    def test_overlapping_names(self):
        # here the 3rd level is different, but the 4th level is the same
        # across the three assignments. this can happen in practice if
        # three different genera are assigned, and under each there is
        # an unnamed species
        # (e.g., f__x;g__A;s__, f__x;g__B;s__, f__x;g__B;s__)
        # in this case, the assignment should be f__x.
        in_ = [['Ab', 'Bc', 'De', 'Jk'],
               ['Ab', 'Bc', 'Fg', 'Jk'],
               ['Ab', 'Bc', 'Hi', 'Jk']]
        actual = _compute_consensus_annotation(in_, 0.51, "Unassigned")
        expected = (['Ab', 'Bc'], 1.)
        self.assertEqual(actual, expected)

        # here the third level is the same in 4/5 of the
        # assignments, but one of them (z, y, c) refers to a
        # different taxa since the higher levels are different.
        # the consensus value should be 3/5, not 4/5, to
        # reflect that.
        in_ = [['a', 'b', 'c'],
               ['a', 'd', 'e'],
               ['a', 'b', 'c'],
               ['a', 'b', 'c'],
               ['z', 'y', 'c']]
        actual = _compute_consensus_annotation(in_, 0.51, "Unassigned")
        expected = (['a', 'b', 'c'], 0.6)
        self.assertEqual(actual, expected)

    def test_adjusts_resolution(self):
        # max result depth is that of shallowest assignment
        in_ = [['Ab', 'Bc', 'Fg'],
               ['Ab', 'Bc', 'Fg', 'Hi'],
               ['Ab', 'Bc', 'Fg', 'Hi'],
               ['Ab', 'Bc', 'Fg', 'Hi'],
               ['Ab', 'Bc', 'Fg', 'Hi', 'Jk']]

        actual = _compute_consensus_annotation(in_, 0.51, "Unassigned")
        expected = (['Ab', 'Bc', 'Fg'], 1.0)
        self.assertEqual(actual, expected)

        in_ = [['Ab', 'Bc', 'Fg'],
               ['Ab', 'Bc', 'Fg', 'Hi', 'Jk'],
               ['Ab', 'Bc', 'Fg', 'Hi', 'Jk'],
               ['Ab', 'Bc', 'Fg', 'Hi', 'Jk'],
               ['Ab', 'Bc', 'Fg', 'Hi', 'Jk']]

        actual = _compute_consensus_annotation(in_, 0.51, "Unassigned")
        expected = (['Ab', 'Bc', 'Fg'], 1.0)
        self.assertEqual(actual, expected)


# This code has been ported from QIIME 1.9.1 with permission from @gregcaporaso
class ConsensusAnnotationsTests(FeatureClassifierTestPluginBase):

    def test_varied_fraction(self):

        in_ = {'q1': [['A', 'B', 'C', 'D'],
                      ['A', 'B', 'C', 'E']],
               'q2': [['A', 'H', 'I', 'J'],
                      ['A', 'H', 'K', 'L', 'M'],
                      ['A', 'H', 'I', 'J']],
               'q3': [[]],
               'q4': [[]],
               'q5': [[]]}
        expected = {'q1': ('A;B;C', 1.0),
                    'q2': ('A;H;I;J', 2. / 3.),
                    'q3': ('Unassigned', 1.0),
                    'q4': ('Unassigned', 1.0),
                    'q5': ('Unassigned', 1.0)}
        actual = _compute_consensus_annotations(in_, 0.51)
        self.assertEqual(actual, expected)

        expected = {'q1': ('A;B;C', 1.0),
                    'q2': ('A;H', 1.0),
                    'q3': ('Unassigned', 1.0),
                    'q4': ('Unassigned', 1.0),
                    'q5': ('Unassigned', 1.0)}
        actual = _compute_consensus_annotations(in_, 0.99)
        self.assertEqual(actual, expected)

    def test_varied_label(self):
        in_ = {'q1': [['A', 'B', 'C', 'D'],
                      ['A', 'B', 'C', 'E']],
               'q2': [['A', 'H', 'I', 'J'],
                      ['A', 'H', 'K', 'L', 'M'],
                      ['A', 'H', 'I', 'J']],
               'q3': [[]],
               'q4': [[]],
               'q5': [[]]}
        expected = {'q1': ('A;B;C', 1.0),
                    'q2': ('A;H;I;J', 2. / 3.),
                    'q3': ('x', 1.0),
                    'q4': ('x', 1.0),
                    'q5': ('x', 1.0)}
        actual = _compute_consensus_annotations(in_, 0.51, "x")
        self.assertEqual(actual, expected)


class OutputNoHitsTests(FeatureClassifierTestPluginBase):

    def test_output_no_hits(self):
        exp = ['>A111', 'ACGTGTGATCGA',
               '>A112', 'ACTGTCATGTGA',
               '>A113', 'ACTGTGTCGTGA']
        obs = {'A111': ('A;B;C;D', 1.0)}
        res = {'A111': ('A;B;C;D', 1.0),
               'A112': ('Unassigned', 0.0),
               'A113': ('Unassigned', 0.0)}
        consensus = _output_no_hits(obs, exp)
        self.assertEqual(consensus, res)
