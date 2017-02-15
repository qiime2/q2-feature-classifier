# ----------------------------------------------------------------------------
# Copyright (c) 2016-2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import tempfile
import subprocess
import pandas as pd
from os.path import isfile
from collections import Counter


def _consensus_assignments(cmd, ref_taxa, t_delim=';', min_consensus=0.51,
                           unassignable_label="Unassigned"):
    '''Run command on CL and find consensus taxonomy.'''
    with tempfile.NamedTemporaryFile() as output:
        cmd = cmd + [output.name]
        _run_command(cmd)
        obs_taxa = _import_blast_assignments(
            output.name, ref_taxa, t_delim=t_delim,
            unassignable_label=unassignable_label)
        consensus = _compute_consensus_annotations(
            obs_taxa, min_consensus=min_consensus, t_delim=t_delim,
            unassignable_label=unassignable_label)
        result = pd.DataFrame.from_dict(consensus, 'index')
        result.index.name = 'Feature ID'
        result.columns = ['Taxon', 'Confidence']
        return result


def _parse_params(str_of_params):
    '''Parse optional parameters, passed as str, to command-line commands.
    This allows additional parameters to be input and passed to the CLI.
    '''
    params = []
    # allow passing of additional BLAST parameters
    if str_of_params is not None:
        params = str_of_params.split(',')
    return params


def _run_command(cmd, verbose=True):
    if verbose:
        print("Running external command line application. This may print "
              "messages to stdout and/or stderr.")
        print("The command being run is below. This command cannot "
              "be manually re-run as it will depend on temporary files that "
              "no longer exist.")
        print("\nCommand:", end=' ')
        print(" ".join(cmd), end='\n\n')
    subprocess.run(cmd, check=True)


def _import_blast_assignments(assignments, ref_taxa, f_delim='\t',
                              t_delim=';', unassignable_label="Unassigned"):
    '''import observed assignments to dict of lists.

    assignments: path or list
        Taxonomy observation map in blast format 6 or 7. Each line consists of
        taxonomy assignments of a query sequence in tab-delimited format:
            <query_id>    <assignment_id>   <...other columns are ignored>

    ref_taxa: dict or pd.Series
        Reference taxonomies in tab-delimited format:
            <accession ID>  kingdom;phylum;class;order;family;genus;species

    f_delim: str
        Field delimiter separating columns in file.

    t_delim: str
        Taxonomy delimiter separating taxonomic levels in taxonomy assignments.
    '''
    obs_taxa = {}

    # accept assignments as list or file
    if isinstance(assignments, list):
        lines = assignments
    elif isfile(assignments):
        with open(assignments, "r") as inputfile:
            lines = [line.strip() for line in inputfile]

    for line in lines:
        if not line.startswith('#') or line == "":
            i = line.split(f_delim)
            # ref taxonomy IDs get imported as a str when using CLI
            # but imported as int in python interpreter (e.g., during tests)
            # the following allows dict lookup as str or int.
            try:
                t = ref_taxa[i[1]].split(t_delim)
            except KeyError:
                try:
                    t = ref_taxa[int(i[1])].split(t_delim)
                # if vsearch fails to find assignment, it reports '*' as the
                # accession ID, which is completely useless and unproductive.
                except ValueError:
                    t = unassignable_label
            if i[0] in obs_taxa.keys():
                obs_taxa[i[0]].append(t)
            else:
                obs_taxa[i[0]] = [t]
    return obs_taxa


def _import_taxonomy_to_dict(infile):
    ''' taxonomy file -> dict'''
    with open(infile, "r") as inputfile:
        lines = {line.strip().split('\t')[0]: line.strip().split('\t')[1]
                 for line in inputfile}
    return lines


def _compute_consensus_annotations(query_annotations,
                                   min_consensus,
                                   t_delim=';',
                                   unassignable_label="Unassigned"):
    """
        Parameters
        ----------
        query_annotations : dict of lists
            Keys are query identifiers, and values are lists of all
            taxonomic annotations associated with that identfier.
        Returns
        -------
        dict
            Keys are query identifiers, and values are the consensus of the
            input taxonomic annotations.
    """
    # This code has been ported and adapted from QIIME 1.9.1 with
    # permission from @gregcaporaso.
    result = {}
    for query_id, annotations in query_annotations.items():
        consensus_annotation, consensus_fraction = \
            _compute_consensus_annotation(annotations, min_consensus,
                                          unassignable_label)
        result[query_id] = (
            t_delim.join(consensus_annotation), consensus_fraction)
    return result


def _compute_consensus_annotation(annotations, min_consensus,
                                  unassignable_label):
    """ Compute the consensus of a collection of annotations
        Parameters
        ----------
        annotations : list of lists
            Taxonomic annotations to compute the consensus of.
        min_consensus : float
            The minimum fraction of the annotations that a specfic annotation
            must be present in for that annotation to be accepted. This must
            be greater than or equal to 0.51.
        unassignable_label : str
            The label to apply if no acceptable annotations are identified.
        Result
        ------
        consensus_annotation
            List containing the consensus assignment
        consensus_fraction
            Fraction of input annotations that agreed at the deepest
            level of assignment
    """
    # This code has been ported from QIIME 1.9.1 with
    # permission from @gregcaporaso.
    if min_consensus <= 0.5:
        raise ValueError("min_consensus must be greater than 0.5.")
    num_input_annotations = len(annotations)
    consensus_annotation = []

    # if the annotations don't all have the same number
    # of levels, the resulting annotation will have a max number
    # of levels equal to the number of levels in the assignment
    # with the fewest number of levels. this is to avoid
    # a case where, for example, there are n assignments, one of
    # which has 7 levels, and the other n-1 assignments have 6 levels.
    # A 7th level in the result would be misleading because it
    # would appear to the user as though it was the consensus
    # across all n assignments.
    num_levels = min([len(a) for a in annotations])

    # iterate over the assignment levels
    for level in range(num_levels):
        # count the different taxonomic assignments at the current level.
        # the counts are computed based on the current level and all higher
        # levels to reflect that, for example, 'p__A; c__B; o__C' and
        # 'p__X; c__Y; o__C' represent different taxa at the o__ level (since
        # they are different at the p__ and c__ levels).
        current_level_annotations = \
            Counter([tuple(e[:level + 1]) for e in annotations])
        # identify the most common taxonomic assignment, and compute the
        # fraction of annotations that contained it. it's safe to compute the
        # fraction using num_assignments because the deepest level we'll
        # ever look at here is num_levels (see above comment on how that
        # is decided).
        tax, max_count = current_level_annotations.most_common(1)[0]
        max_consensus_fraction = max_count / num_input_annotations
        # check whether the most common taxonomic assignment is observed
        # in at least min_consensus of the sequences
        if max_consensus_fraction >= min_consensus:
            # if so, append the current level only (e.g., 'o__C' if tax is
            # 'p__A; c__B; o__C', and continue on to the next level
            consensus_annotation.append((tax[-1], max_consensus_fraction))
        else:
            # if not, there is no assignment at this level, and we're
            # done iterating over levels
            break

    # construct the results
    # determine the number of levels in the consensus assignment
    consensus_annotation_depth = len(consensus_annotation)
    if consensus_annotation_depth > 0:
        # if it's greater than 0, generate a list of the
        # taxa assignments at each level
        annotation = [a[0] for a in consensus_annotation]
        # and assign the consensus_fraction_result as the
        # consensus fraction at the deepest level
        consensus_fraction_result = \
            consensus_annotation[consensus_annotation_depth - 1][1]
    else:
        # if there are zero assignments, indicate that the taxa is
        # unknown
        annotation = [unassignable_label]
        # and assign the consensus_fraction_result to 1.0 (this is
        # somewhat arbitrary, but could be interpreted as all of the
        # assignments suggest an unknown taxonomy)
        consensus_fraction_result = 1.0

    return annotation, consensus_fraction_result
