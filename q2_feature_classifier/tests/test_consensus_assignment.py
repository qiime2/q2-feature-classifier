# ----------------------------------------------------------------------------
# Copyright (c) 2016-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import pandas as pd
import pandas.testing as pdt

from qiime2.sdk import Artifact
from q2_feature_classifier._skl import _specific_fitters
from q2_feature_classifier._consensus_assignment import (
    _lca_consensus,
    _compute_consensus_annotations,
    _blast6format_df_to_series_of_lists,
    _taxa_to_cumulative_ranks)
from q2_types.feature_data import DNAFASTAFormat
from . import FeatureClassifierTestPluginBase
from qiime2.plugins import feature_classifier as qfc


class SequenceSearchTests(FeatureClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        self.query = Artifact.import_data(
            'FeatureData[Sequence]', self.get_data_path('query-seqs.fasta'))
        self.ref = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))

    def test_blast(self):
        result, = qfc.actions.blast(
            self.query, self.ref, maxaccepts=3, perc_identity=0.9)
        exp = pd.DataFrame({
            'qseqid': {0: '1111561', 1: '1111561', 2: '1111561', 3: '835097',
                       4: 'junk'},
            'sseqid': {0: '1111561', 1: '574274', 2: '149351', 3: '835097',
                       4: '*'},
            'pident': {0: 100.0, 1: 92.308, 2: 91.781, 3: 100.0, 4: 0.0},
            'length': {0: 75.0, 1: 78.0, 2: 73.0, 3: 80.0, 4: 0.0},
            'mismatch': {0: 0.0, 1: 2.0, 2: 4.0, 3: 0.0, 4: 0.0},
            'gapopen': {0: 0.0, 1: 4.0, 2: 2.0, 3: 0.0, 4: 0.0},
            'qstart': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
            'qend': {0: 75.0, 1: 75.0, 2: 71.0, 3: 80.0, 4: 0.0},
            'sstart': {0: 24.0, 1: 24.0, 2: 24.0, 3: 32.0, 4: 0.0},
            'send': {0: 98.0, 1: 100.0, 2: 96.0, 3: 111.0, 4: 0.0},
            'evalue': {0: 8.35e-36, 1: 2.36e-26, 2: 3.94e-24,
                       3: 1.5000000000000002e-38, 4: 0.0},
            'bitscore': {0: 139.0, 1: 108.0, 2: 100.0, 3: 148.0, 4: 0.0}})
        pdt.assert_frame_equal(result.view(pd.DataFrame), exp)

    def test_blast_no_output_no_hits(self):
        result, = qfc.actions.blast(
            self.query, self.ref, maxaccepts=3, perc_identity=0.9,
            output_no_hits=False)
        exp = pd.DataFrame({
            'qseqid': {0: '1111561', 1: '1111561', 2: '1111561', 3: '835097'},
            'sseqid': {0: '1111561', 1: '574274', 2: '149351', 3: '835097'},
            'pident': {0: 100.0, 1: 92.308, 2: 91.781, 3: 100.0},
            'length': {0: 75.0, 1: 78.0, 2: 73.0, 3: 80.0},
            'mismatch': {0: 0.0, 1: 2.0, 2: 4.0, 3: 0.0},
            'gapopen': {0: 0.0, 1: 4.0, 2: 2.0, 3: 0.0},
            'qstart': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            'qend': {0: 75.0, 1: 75.0, 2: 71.0, 3: 80.0},
            'sstart': {0: 24.0, 1: 24.0, 2: 24.0, 3: 32.0},
            'send': {0: 98.0, 1: 100.0, 2: 96.0, 3: 111.0},
            'evalue': {0: 8.35e-36, 1: 2.36e-26, 2: 3.94e-24,
                       3: 1.5000000000000002e-38},
            'bitscore': {0: 139.0, 1: 108.0, 2: 100.0, 3: 148.0}})
        pdt.assert_frame_equal(result.view(pd.DataFrame), exp)

    def test_vsearch_global(self):
        result, = qfc.actions.vsearch_global(
            self.query, self.ref, maxaccepts=3, perc_identity=0.9)
        exp = pd.DataFrame({
            'qseqid': {0: '1111561', 1: '835097', 2: 'junk'},
            'sseqid': {0: '1111561', 1: '835097', 2: '*'},
            'pident': {0: 100.0, 1: 100.0, 2: 0.0},
            'length': {0: 75.0, 1: 80.0, 2: 0.0},
            'mismatch': {0: 0.0, 1: 0.0, 2: 0.0},
            'gapopen': {0: 0.0, 1: 0.0, 2: 0.0},
            'qstart': {0: 1.0, 1: 1.0, 2: 0.0},
            'qend': {0: 75.0, 1: 80.0, 2: 0.0},
            'sstart': {0: 1.0, 1: 1.0, 2: 0.0},
            'send': {0: 150.0, 1: 150.0, 2: 0.0},
            'evalue': {0: -1.0, 1: -1.0, 2: -1.0},
            'bitscore': {0: 0.0, 1: 0.0, 2: 0.0}})
        pdt.assert_frame_equal(
            result.view(pd.DataFrame), exp, check_names=False)

    def test_vsearch_global_no_output_no_hits(self):
        result, = qfc.actions.vsearch_global(
            self.query, self.ref, maxaccepts=3, perc_identity=0.9,
            output_no_hits=False)
        exp = pd.DataFrame({
            'qseqid': {0: '1111561', 1: '835097'},
            'sseqid': {0: '1111561', 1: '835097'},
            'pident': {0: 100.0, 1: 100.0},
            'length': {0: 75.0, 1: 80.0},
            'mismatch': {0: 0.0, 1: 0.0},
            'gapopen': {0: 0.0, 1: 0.0},
            'qstart': {0: 1.0, 1: 1.0},
            'qend': {0: 75.0, 1: 80.0},
            'sstart': {0: 1.0, 1: 1.0},
            'send': {0: 150.0, 1: 150.0},
            'evalue': {0: -1.0, 1: -1.0},
            'bitscore': {0: 0.0, 1: 0.0}})
        pdt.assert_frame_equal(
            result.view(pd.DataFrame), exp, check_names=False)

    def test_vsearch_global_permissive(self):
        result, = qfc.actions.vsearch_global(
            self.query, self.ref, maxaccepts=1, perc_identity=0.8,
            query_cov=0.2)
        exp = pd.DataFrame({
            'qseqid': {0: '1111561', 1: '835097', 2: 'junk'},
            'sseqid': {0: '1111561', 1: '835097', 2: '4314518'},
            'pident': {0: 100.0, 1: 100.0, 2: 90.0},
            'length': {0: 75.0, 1: 80.0, 2: 20.0},
            'mismatch': {0: 0.0, 1: 0.0, 2: 2.0},
            'gapopen': {0: 0.0, 1: 0.0, 2: 0.0},
            'qstart': {0: 1.0, 1: 1.0, 2: 1.0},
            'qend': {0: 75.0, 1: 80.0, 2: 100.0},
            'sstart': {0: 1.0, 1: 1.0, 2: 1.0},
            'send': {0: 150.0, 1: 150.0, 2: 95.0},
            'evalue': {0: -1.0, 1: -1.0, 2: -1.0},
            'bitscore': {0: 0.0, 1: 0.0, 2: 0.0}})
        pdt.assert_frame_equal(
            result.view(pd.DataFrame), exp, check_names=False)


# setting up utility test for comparing series below
def series_is_subset(expected, observed):
    # join observed and expected results to compare
    joined = pd.concat([expected, observed], axis=1, join='inner')
    # check that all observed results are at least a substring of expected
    # (this should usually be the case, unless if consensus classification
    # did very badly, e.g., resulting in unclassified)
    compared = joined.apply(lambda x: x[0].startswith(x[1]), axis=1)
    # in the original tests we set a threshold of 50% for subsets... most
    # should be but in some cases misclassification could occur, or dodgy
    # annotations that screw up the LCA. So just check that we have at least
    # as many             TRUE     as                 FALSE.
    return len(compared[compared]) >= len(compared[~compared])


class ConsensusAssignmentsTests(FeatureClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        self.taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', self.get_data_path('taxonomy.tsv'))
        self.reads = Artifact.import_data(
            'FeatureData[Sequence]',
            self.get_data_path('se-dna-sequences.fasta'))
        self.exp = self.taxonomy.view(pd.Series)

    # Make sure blast and vsearch produce expected outputs
    # but there is no "right" taxonomy assignment.
    # TODO: the results should be deterministic, so we should check expected
    # search and/or taxonomy classification outputs.
    def test_classify_consensus_blast(self):
        result, _, = qfc.actions.classify_consensus_blast(
            self.reads, self.reads, self.taxonomy)
        self.assertTrue(series_is_subset(self.exp, result.view(pd.Series)))

    def test_classify_consensus_vsearch(self):
        result, _, = qfc.actions.classify_consensus_vsearch(
            self.reads, self.reads, self.taxonomy)
        self.assertTrue(series_is_subset(self.exp, result.view(pd.Series)))

    # search_exact with all other exposed params to confirm compatibility
    # in future releases of vsearch
    def test_classify_consensus_vsearch_search_exact(self):
        result, _, = qfc.actions.classify_consensus_vsearch(
            self.reads, self.reads, self.taxonomy, search_exact=True,
            top_hits_only=True, output_no_hits=True, weak_id=0.9, maxhits=10)
        self.assertTrue(series_is_subset(self.exp, result.view(pd.Series)))

    def test_classify_consensus_vsearch_top_hits_only(self):
        result, _, = qfc.actions.classify_consensus_vsearch(
            self.reads, self.reads, self.taxonomy, top_hits_only=True)
        self.assertTrue(series_is_subset(self.exp, result.view(pd.Series)))

    # make sure weak_id and other parameters do not conflict with each other.
    # This test just makes sure the command runs okay with all options.
    # We are not in the business of debugging VSEARCH, but want to have this
    # test as a canary in the coal mine.
    def test_classify_consensus_vsearch_the_works(self):
        result, _, = qfc.actions.classify_consensus_vsearch(
            self.reads, self.reads, self.taxonomy, top_hits_only=True,
            maxhits=1, maxrejects=10, weak_id=0.8, perc_identity=0.99,
            output_no_hits=False)
        self.assertTrue(series_is_subset(self.exp, result.view(pd.Series)))


class HybridClassiferTests(FeatureClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', self.get_data_path('taxonomy.tsv'))
        self.taxonomy = taxonomy.view(pd.Series)
        self.taxartifact = taxonomy
        # TODO: use `Artifact.import_data` here once we have a transformer
        # for DNASequencesDirectoryFormat -> DNAFASTAFormat
        reads_fp = self.get_data_path('se-dna-sequences.fasta')
        reads = DNAFASTAFormat(reads_fp, mode='r')
        self.reads = Artifact.import_data('FeatureData[Sequence]', reads)

        fitter = getattr(qfc.methods,
                         'fit_classifier_' + _specific_fitters[0][0])
        self.classifier = fitter(self.reads, self.taxartifact).classifier

        self.query = Artifact.import_data('FeatureData[Sequence]', pd.Series(
            {'A': 'GCCTAACACATGCAAGTCGAACGGCAGCGGGGGAAAGCTTGCTTTCCTGCCGGCGA',
             'B': 'TAACACATGCAAGTCAACGATGCTTATGTAGCAATATGTAAGTAGAGTGGCGCACG',
             'C': 'ATACATGCAAGTCGTACGGTATTCCGGTTTCGGCCGGGAGAGAGTGGCGGATGGGT',
             'D': 'GACGAACGCTGGCGACGTGCTTAACACATGCAAGTCGTGCGAGGACGGGCGGTGCT'
                  'TGCACTGCTCGAGCCGAGCGGCGGACGGGTGAGTAACACGTGAGCAACCTATCTCC'
                  'GTGCGGGGGACAACCCGGGGAAACCCGGGCTAATACCG'}))

    def test_classify_hybrid_vsearch_sklearn_all_exact_match(self):

        result, = qfc.actions.classify_hybrid_vsearch_sklearn(
            query=self.reads, reference_reads=self.reads,
            reference_taxonomy=self.taxartifact, classifier=self.classifier,
            prefilter=False)
        result, = qfc.actions.classify_hybrid_vsearch_sklearn(
            query=self.reads, reference_reads=self.reads,
            reference_taxonomy=self.taxartifact, classifier=self.classifier)
        result = result.view(pd.DataFrame)
        res = result.Taxon.to_dict()
        tax = self.taxonomy.to_dict()
        right = 0.
        for taxon in res:
            right += tax[taxon].startswith(res[taxon])
        self.assertGreater(right/len(res), 0.5)

    def test_classify_hybrid_vsearch_sklearn_mixed_query(self):

        result, = qfc.actions.classify_hybrid_vsearch_sklearn(
            query=self.query, reference_reads=self.reads,
            reference_taxonomy=self.taxartifact, classifier=self.classifier,
            prefilter=True, read_orientation='same', randseed=1001)
        result = result.view(pd.DataFrame)
        obs = result.Taxon.to_dict()
        exp = {'A': 'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; '
                    'o__Legionellales; f__; g__; s__',
               'B': 'k__Bacteria; p__Chlorobi; c__; o__; f__; g__; s__',
               'C': 'k__Bacteria; p__Bacteroidetes; c__Cytophagia; '
                    'o__Cytophagales; f__Cyclobacteriaceae; g__; s__',
               'D': 'k__Bacteria; p__Gemmatimonadetes; c__Gemm-5; o__; f__; '
                    'g__; s__'}
        self.assertDictEqual(obs, exp)


class ImportBlastAssignmentTests(FeatureClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        result = Artifact.import_data(
            'FeatureData[BLAST6]', self.get_data_path('blast6-format.tsv'))
        self.result = result.view(pd.DataFrame)
        taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', self.get_data_path('taxonomy.tsv'))
        self.taxonomy = taxonomy.view(pd.Series)

    def test_blast6format_df_to_series_of_lists(self):
        # and add in a query without any hits, to check that it is parsed
        self.result.loc[3] = ['junk', '*'] + [''] * 10
        obs = _blast6format_df_to_series_of_lists(self.result, self.taxonomy)
        exp = pd.Series(
            {'1111561': [
                'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; '
                'o__Legionellales; f__; g__; s__',
                'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; '
                'o__Legionellales; f__Coxiellaceae; g__; s__'],
             '835097': [
                'k__Bacteria; p__Chloroflexi; c__SAR202; o__; f__; g__; s__'],
             'junk': ['Unassigned']},
            name='sseqid')
        exp.index.name = 'qseqid'
        pdt.assert_series_equal(exp, obs)

    # should fail when hit IDs are missing from reference taxonomy
    # in this case 1128818 is missing
    def test_blast6format_df_to_series_of_lists_fail_on_missing_ids(self):
        # add a bad idea
        self.result.loc[3] = ['junk', 'lost-id'] + [''] * 10
        with self.assertRaisesRegex(KeyError, "results do not match.*lost-id"):
            _blast6format_df_to_series_of_lists(self.result, self.taxonomy)


class ConsensusAnnotationTests(FeatureClassifierTestPluginBase):

    def test_taxa_to_cumulative_ranks(self):
        taxa = ['a;b;c', 'a;b;d', 'a;g;g']
        exp = [['a', 'a;b', 'a;b;c'], ['a', 'a;b', 'a;b;d'],
               ['a', 'a;g', 'a;g;g']]
        self.assertEqual(_taxa_to_cumulative_ranks(taxa), exp)

    def test_taxa_to_cumulative_ranks_with_uneven_ranks(self):
        taxa = ['a;b;c', 'a;b;d', 'a;g;g;somemoregarbage']
        exp = [['a', 'a;b', 'a;b;c'], ['a', 'a;b', 'a;b;d'],
               ['a', 'a;g', 'a;g;g', 'a;g;g;somemoregarbage']]
        self.assertEqual(_taxa_to_cumulative_ranks(taxa), exp)

    def test_taxa_to_cumulative_ranks_with_one_entry(self):
        taxa = ['a;b;c']
        exp = [['a', 'a;b', 'a;b;c']]
        self.assertEqual(_taxa_to_cumulative_ranks(taxa), exp)

    def test_taxa_to_cumulative_ranks_with_empty_list(self):
        taxa = ['']
        exp = [['']]
        self.assertEqual(_taxa_to_cumulative_ranks(taxa), exp)

    def test_varied_min_fraction(self):
        in_ = [['Ab', 'Ab;Bc', 'Ab;Bc;De'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Jk']]

        actual = _lca_consensus(in_, 0.51, "Unassigned")
        expected = ('Ab;Bc;Fg', 0.667)
        self.assertEqual(actual, expected)

        # increased min_consensus_fraction yields decreased specificity
        actual = _lca_consensus(in_, 0.99, "Unassigned")
        expected = ('Ab;Bc', 1.0)
        self.assertEqual(actual, expected)

    def test_single_annotation(self):
        in_ = [['Ab', 'Ab;Bc', 'Ab;Bc;De']]

        actual = _lca_consensus(in_, 1.0, "Unassigned")
        expected = ('Ab;Bc;De', 1.0)
        self.assertEqual(actual, expected)

        actual = _lca_consensus(in_, 0.501, "Unassigned")
        expected = ('Ab;Bc;De', 1.0)
        self.assertEqual(actual, expected)

    def test_no_consensus(self):
        in_ = [['Ab', 'Ab;Bc', 'Ab;Bc;De'],
               ['Cd', 'Cd;Bc', 'Cd;Bc;Fg', 'Cd;Bc;Fg;Hi'],
               ['Ef', 'Ef;Bc', 'Ef;Bc;Fg', 'Ef;Bc;Fg;Jk']]

        actual = _lca_consensus(in_, 0.51, "Unassigned")
        expected = ('Unassigned', 0.)
        self.assertEqual(actual, expected)

        actual = _lca_consensus(
                    in_, 0.51, unassignable_label="Hello world!")
        expected = ('Hello world!', 0.)
        self.assertEqual(actual, expected)

    def test_overlapping_names(self):
        # here the 3rd level is different, but the 4th level is the same
        # across the three assignments. this can happen in practice if
        # three different genera are assigned, and under each there is
        # an unnamed species
        # (e.g., f__x;g__A;s__, f__x;g__B;s__, f__x;g__B;s__)
        # in this case, the assignment should be f__x.
        in_ = [['Ab', 'Ab;Bc', 'Ab;Bc;De', 'Ab;Bc;De;Jk'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Jk'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Hi', 'Ab;Bc;Hi;Jk']]
        actual = _lca_consensus(in_, 0.51, "Unassigned")
        expected = ('Ab;Bc', 1.)
        self.assertEqual(actual, expected)

        # here the third level is the same in 4/5 of the
        # assignments, but one of them (z, y, c) refers to a
        # different taxa since the higher levels are different.
        # the consensus value should be 3/5, not 4/5, to
        # reflect that.
        in_ = [['a', 'a;b', 'a;b;c'],
               ['a', 'a;d', 'a;d;e'],
               ['a', 'a;b', 'a;b;c'],
               ['a', 'a;b', 'a;b;c'],
               ['z', 'z;y', 'z;y;c']]
        actual = _lca_consensus(in_, 0.51, "Unassigned")
        expected = ('a;b;c', 0.6)
        self.assertEqual(actual, expected)

    def test_adjusts_resolution(self):
        # max result depth is that of shallowest assignment
        # Reading this test now, I am not entirely sure that this is how
        # such cases should be handled. Technically such a case should not
        # arise (as the dbs should have even ranks) so we can leave this for
        # now, and it is arguable, but in this case I think that majority
        # should rule. We use `zip` but might want to consider `zip_longest`.
        in_ = [['Ab', 'Ab;Bc', 'Ab;Bc;Fg'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi', 'Ab;Bc;Fg;Hi;Jk']]

        actual = _lca_consensus(in_, 0.51, "Unassigned")
        expected = ('Ab;Bc;Fg', 1.0)
        self.assertEqual(actual, expected)

        in_ = [['Ab', 'Ab;Bc', 'Ab;Bc;Fg'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi', 'Ab;Bc;Fg;Hi;Jk'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi', 'Ab;Bc;Fg;Hi;Jk'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi', 'Ab;Bc;Fg;Hi;Jk'],
               ['Ab', 'Ab;Bc', 'Ab;Bc;Fg', 'Ab;Bc;Fg;Hi', 'Ab;Bc;Fg;Hi;Jk']]

        actual = _lca_consensus(in_, 0.51, "Unassigned")
        expected = ('Ab;Bc;Fg', 1.0)
        self.assertEqual(actual, expected)


# More edge cases are tested for the internals above, so the tests here are
# made slim to just test the overarching functions.
class ConsensusAnnotationsTests(FeatureClassifierTestPluginBase):

    def test_varied_fraction(self):

        in_ = pd.Series({'q1': ['A;B;C;D', 'A;B;C;E'],
                         'q2': ['A;H;I;J', 'A;H;K;L;M', 'A;H;I;J'],
                         'q3': ['A', 'A', 'B'],
                         'q4': ['A', 'B'],
                         'q5': []})
        expected = pd.DataFrame({
            'Taxon': {'q1': 'A;B;C', 'q2': 'A;H;I;J', 'q3': 'A',
                      'q4': 'Unassigned', 'q5': 'Unassigned'},
            'Consensus': {
                'q1': 1.0, 'q2': 0.667, 'q3': 0.667, 'q4': 0.0, 'q5': 0.0}})
        actual = _compute_consensus_annotations(in_, 0.51, 'Unassigned')
        pdt.assert_frame_equal(actual, expected, check_names=False)

        expected = pd.DataFrame({
            'Taxon': {'q1': 'A;B;C', 'q2': 'A;H', 'q3': 'Unassigned',
                      'q4': 'Unassigned', 'q5': 'Unassigned'},
            'Consensus': {
                'q1': 1.0, 'q2': 1.0, 'q3': 0.0, 'q4': 0.0, 'q5': 0.0}})
        actual = _compute_consensus_annotations(in_, 0.99, 'Unassigned')
        pdt.assert_frame_equal(actual, expected, check_names=False)

    def test_find_consensus_annotation(self):

        result = Artifact.import_data(
            'FeatureData[BLAST6]', self.get_data_path('blast6-format.tsv'))
        taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', self.get_data_path('taxonomy.tsv'))
        consensus, = qfc.actions.find_consensus_annotation(result, taxonomy)
        obs = consensus.view(pd.DataFrame)
        exp = pd.DataFrame(
            {'Taxon': {
                '1111561': 'k__Bacteria; p__Proteobacteria; '
                           'c__Gammaproteobacteria; o__Legionellales',
                '835097': 'k__Bacteria; p__Chloroflexi; c__SAR202; o__; f__; '
                          'g__; s__'},
             'Consensus': {'1111561': '1.0', '835097': '1.0'}})
        pdt.assert_frame_equal(exp, obs, check_names=False)
