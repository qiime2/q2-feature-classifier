# ----------------------------------------------------------------------------
# Copyright (c) 2016-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from collections import Counter
from math import ceil

import pandas as pd

from qiime2.plugin import Str, Float, Range
from .plugin_setup import plugin
from q2_types.feature_data import FeatureData, Taxonomy, BLAST6


min_consensus_param = {'min_consensus': Float % Range(
    0.5, 1.0, inclusive_end=True, inclusive_start=False)}

min_consensus_param_description = {
    'min_consensus': 'Minimum fraction of assignments must match top '
                     'hit to be accepted as consensus assignment.'}

DEFAULTUNASSIGNABLELABEL = "Unassigned"


def find_consensus_annotation(search_results: pd.DataFrame,
                              reference_taxonomy: pd.Series,
                              min_consensus: int = 0.51,
                              unassignable_label: str =
                              DEFAULTUNASSIGNABLELABEL
                              ) -> pd.DataFrame:
    """Find consensus taxonomy from BLAST6Format alignment summary.

    search_results: pd.dataframe
        BLAST6Format search results with canonical headers attached.
    reference_taxonomy: pd.Series
        Annotations of reference database used for original search.
    min_consensus : float
        The minimum fraction of the annotations that a specific annotation
        must be present in for that annotation to be accepted. Current
        lower boundary is 0.51.
    unassignable_label : str
        The label to apply if no acceptable annotations are identified.
    """
    # load and convert blast6format results to dict of taxa hits
    obs_taxa = _blast6format_df_to_series_of_lists(
        search_results, reference_taxonomy,
        unassignable_label=unassignable_label)
    # TODO: is it worth allowing early stopping if maxaccepts==1?
    # compute consensus annotations
    result = _compute_consensus_annotations(
        obs_taxa, min_consensus=min_consensus,
        unassignable_label=unassignable_label)
    result.index.name = 'Feature ID'
    return result


plugin.methods.register_function(
    function=find_consensus_annotation,
    inputs={'search_results': FeatureData[BLAST6],
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={
        **min_consensus_param,
        'unassignable_label': Str},
    outputs=[('consensus_taxonomy', FeatureData[Taxonomy])],
    input_descriptions={
        'search_results': 'Search results in BLAST6 output format',
        'reference_taxonomy': 'reference taxonomy labels.'},
    parameter_descriptions={
        **min_consensus_param_description,
        'unassignable_label': 'Annotation given when no consensus is found.'
    },
    output_descriptions={
        'consensus_taxonomy': 'Consensus taxonomy and scores.'},
    name='Find consensus among multiple annotations.',
    description=('Find consensus annotation for each query searched against '
                 'a reference database, by finding the least common ancestor '
                 'among one or more semicolon-delimited hierarchical '
                 'annotations. Note that the annotation hierarchy is assumed '
                 'to have an even number of ranks.'),
)


def _blast6format_df_to_series_of_lists(
        assignments: pd.DataFrame,
        ref_taxa: pd.Series,
        unassignable_label: str = DEFAULTUNASSIGNABLELABEL
) -> pd.Series:
    """import observed assignments in blast6 format to series of lists.

    assignments: pd.DataFrame
        Taxonomy observation map in blast format 6. Each line consists of
        taxonomy assignments of a query sequence in tab-delimited format:
            <query_id>    <subject-seq-id>   <...other columns are ignored>

    ref_taxa: pd.Series
        Reference taxonomies in tab-delimited format:
            <accession ID>  Annotation
        The accession IDs in this taxonomy should match the subject-seq-ids in
        the "assignment" input.
    """
    # validate that assignments are present in reference taxonomy
    # (i.e., that the correct reference taxonomy was used).
    # Note that we drop unassigned labels from this set.
    missing_ids = \
        set(assignments['sseqid'].values) - set(ref_taxa.index) - {'*', ''}
    if len(missing_ids) > 0:
        raise KeyError('Reference taxonomy and search results do not match. '
                       'The following identifiers were reported in the search '
                       'results but are not present in the reference taxonomy:'
                       ' {0}'.format(', '.join(str(i) for i in missing_ids)))

    # if vsearch fails to find assignment, it reports '*' as the
    # accession ID, so we will add this mapping to the reference taxonomy.
    ref_taxa['*'] = unassignable_label
    assignments_copy = assignments.copy(deep=True)
    for index, value in assignments_copy.iterrows():
        sseqid = assignments_copy.iloc[index]['sseqid']
        assignments_copy.at[index, 'sseqid'] = ref_taxa.at[sseqid]
    # convert to dict of {accession_id: [annotations]}
    taxa_hits: pd.Series = assignments_copy.set_index('qseqid')['sseqid']
    taxa_hits = taxa_hits.groupby(taxa_hits.index).apply(list)

    return taxa_hits


def _compute_consensus_annotations(
        query_annotations, min_consensus,
        unassignable_label=DEFAULTUNASSIGNABLELABEL):
    """
        Parameters
        ----------
        query_annotations : pd.Series of lists
            Indices are query identifiers, and values are lists of all
            taxonomic annotations associated with that identifier.
        Returns
        -------
        pd.DataFrame
            Indices are query identifiers, and values are the consensus of the
            input taxonomic annotations, and the consensus score.
    """
    # define function to apply to each list of taxa hits
    # Note: I am setting this up to open the possibility to define other
    # functions later (e.g., not simple threshold consensus)
    def _apply_consensus_function(taxa, min_consensus=min_consensus,
                                  unassignable_label=unassignable_label,
                                  _consensus_function=_lca_consensus):
        # if there is no consensus, skip consensus calculation
        if len(taxa) == 1:
            taxa, score = taxa.pop(), 1.
        else:
            taxa = _taxa_to_cumulative_ranks(taxa)
            # apply and score consensus
            taxa, score = _consensus_function(
                taxa, min_consensus, unassignable_label)
        # return as a series so that the outer apply returns a dataframe
        # (i.e., consensus scores get inserted as an additional column)
        return pd.Series([taxa, score], index=['Taxon', 'Consensus'])

    # If func returns a Series object the result will be a DataFrame.
    return query_annotations.apply(_apply_consensus_function)


# first split semicolon-delimited taxonomies by rank
# and iteratively join ranks, so that: ['a;b;c', 'a;b;d', 'a;g;g'] -->
# [['a', 'a;b', 'a;b;c'], ['a', 'a;b', 'a;b;d'], ['a', 'a;g', 'a;g;g']]
# this is to avoid issues where e.g., the same species name may occur
# in different taxonomic lineages.
def _taxa_to_cumulative_ranks(taxa):
    """
        Parameters
        ----------
        taxa : list or str
            List of semicolon-delimited taxonomic labels.
            e.g., ['a;b;c', 'a;b;d']
        Returns
        -------
        list of lists of str
            Lists of cumulative taxonomic ranks for each input str
            e.g., [['a', 'a;b', 'a;b;c'], ['a', 'a;b', 'a;b;d']]
    """
    return [[';'.join(t.split(';')[:n + 1])
             for n in range(t.count(';') + 1)]
            for t in taxa]


# Find the LCA by consensus threshold. Return label and the consensus score.
def _lca_consensus(annotations, min_consensus, unassignable_label):
    """ Compute the consensus of a collection of annotations
        Parameters
        ----------
        annotations : list of lists
            Taxonomic annotations to form consensus.
        min_consensus : float
            The minimum fraction of the annotations that a specific annotation
            must be present in for that annotation to be accepted. Current
            lower boundary is 0.51.
        unassignable_label : str
            The label to apply if no acceptable annotations are identified.
        Result
        ------
        consensus_annotation: str
            The consensus assignment
        consensus_fraction: float
            Fraction of input annotations that agreed at the deepest
            level of assignment
    """
    # count total number of labels to get consensus threshold
    n_annotations = len(annotations)
    threshold = ceil(n_annotations * min_consensus)
    # zip together ranks and count frequency of each unique label.
    # This assumes that a hierarchical taxonomy with even numbers of
    # ranks was used.
    taxa_comparison = [Counter(rank) for rank in zip(*annotations)]
    # iterate rank comparisons in reverse
    # to find rank with consensus count > threshold
    for rank in taxa_comparison[::-1]:
        # grab most common label and its count
        label, count = rank.most_common(1)[0]
        # TODO: this assumes that min_consensus >= 0.51 (current lower bound)
        # but could fail to find ties if we allow lower min_consensus scores
        if count >= threshold:
            return label, round(count / n_annotations, 3)
    # if we reach this point, no consensus was ever found at any rank
    return unassignable_label, 0.0
