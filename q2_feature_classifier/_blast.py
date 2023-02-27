# ----------------------------------------------------------------------------
# Copyright (c) 2016-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import subprocess
import pandas as pd
from q2_types.feature_data import (
    FeatureData, Taxonomy, Sequence, DNAFASTAFormat, DNAIterator, BLAST6,
    BLAST6Format)
from qiime2.plugin import Int, Str, Float, Choices, Range, Bool
from .plugin_setup import plugin, citations
from ._consensus_assignment import (
    min_consensus_param, min_consensus_param_description,
    DEFAULTUNASSIGNABLELABEL)

# ---------------------------------------------------------------
# Reason for num_thread not being exposed.
# BLAST doesn't allow threading when a subject is provided(As of 2/19/20).
# num_thread was removed to prevent warning that stated:
# "'Num_thread' is currently ignored when 'subject' is specified"(issue #77).
# Seen here: https://github.com/qiime2/q2-feature-classifier/issues/77.
# A '-subject' input is required in this function.
# Therefore num_thread is not exposable.
# ---------------------------------------------------------------


# Specify default settings for various functions
DEFAULTMAXACCEPTS = 10
DEFAULTPERCENTID = 0.8
DEFAULTQUERYCOV = 0.8
DEFAULTSTRAND = 'both'
DEFAULTEVALUE = 0.001
DEFAULTMINCONSENSUS = 0.51
DEFAULTOUTPUTNOHITS = True


# NOTE FOR THE FUTURE: should this be called blastn? would it be possible to
# eventually generalize to e.g., blastp or blastx? or will this be too
# challenging, e.g., to detect the format internally? A `mode` parameter could
# be added and TypeMapped to the input type, a bit cumbersome but one way to
# accomplish this without code bloat. But the question is: would we want to
# expose different parameters etc? My feeling is let's call this `blast` for
# now and then cross that bridge when we come to it.
def blast(query: DNAFASTAFormat,
          reference_reads: DNAFASTAFormat,
          maxaccepts: int = DEFAULTMAXACCEPTS,
          perc_identity: float = DEFAULTPERCENTID,
          query_cov: float = DEFAULTQUERYCOV,
          strand: str = DEFAULTSTRAND,
          evalue: float = DEFAULTEVALUE,
          output_no_hits: bool = DEFAULTOUTPUTNOHITS) -> pd.DataFrame:
    perc_identity = perc_identity * 100
    query_cov = query_cov * 100
    seqs_fp = str(query)
    ref_fp = str(reference_reads)
    # TODO: generalize to support other blast types?
    output = BLAST6Format()
    cmd = ['blastn', '-query', seqs_fp, '-evalue', str(evalue), '-strand',
           strand, '-outfmt', '6', '-subject', ref_fp, '-perc_identity',
           str(perc_identity), '-qcov_hsp_perc', str(query_cov),
           '-max_target_seqs', str(maxaccepts), '-out', str(output)]
    _run_command(cmd)
    # load as dataframe to quickly validate (note: will fail now if empty)
    result = output.view(pd.DataFrame)

    # blast will not report reads with no hits. We will report this
    # information here, so that it is explicitly reported to the user.
    if output_no_hits:
        ids_with_hit = set(result['qseqid'].unique())
        query_ids = {seq.metadata['id'] for seq in query.view(DNAIterator)}
        missing_ids = query_ids - ids_with_hit
        if len(missing_ids) > 0:
            # we will mirror vsearch behavior and annotate unassigneds as '*'
            # and fill other columns with 0 values (np.nan makes format error).
            missed = pd.DataFrame(columns=result.columns)
            missed = missed.append(
                [{'qseqid': i, 'sseqid': '*'} for i in missing_ids]).fillna(0)
            # Do two separate appends to make sure that fillna does not alter
            # any other contents from the original search results.
            result = result.append(missed, ignore_index=True)
    return result


def classify_consensus_blast(ctx,
                             query,
                             reference_reads,
                             reference_taxonomy,
                             maxaccepts=DEFAULTMAXACCEPTS,
                             perc_identity=DEFAULTPERCENTID,
                             query_cov=DEFAULTQUERYCOV,
                             strand=DEFAULTSTRAND,
                             evalue=DEFAULTEVALUE,
                             output_no_hits=DEFAULTOUTPUTNOHITS,
                             min_consensus=DEFAULTMINCONSENSUS,
                             unassignable_label=DEFAULTUNASSIGNABLELABEL):
    search_db = ctx.get_action('feature_classifier', 'blast')
    lca = ctx.get_action('feature_classifier', 'find_consensus_annotation')
    result, = search_db(query=query, reference_reads=reference_reads,
                        maxaccepts=maxaccepts, perc_identity=perc_identity,
                        query_cov=query_cov, strand=strand, evalue=evalue,
                        output_no_hits=output_no_hits)
    consensus, = lca(search_results=result,
                     reference_taxonomy=reference_taxonomy,
                     min_consensus=min_consensus,
                     unassignable_label=unassignable_label)
    # New: add BLAST6Format result as an output. This could just as well be a
    # visualizer generated from these results (using q2-metadata tabulate).
    # Would that be more useful to the user that the QZA?
    return consensus, result


# Replace this function with QIIME2 API for wrapping commands/binaries,
# pending https://github.com/qiime2/qiime2/issues/224
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


inputs = {'query': FeatureData[Sequence],
          'reference_reads': FeatureData[Sequence]}

input_descriptions = {'query': 'Query sequences.',
                      'reference_reads': 'Reference sequences.'}

classification_output = ('classification', FeatureData[Taxonomy])

classification_output_description = {
    'classification': 'Taxonomy classifications of query sequences.'}

parameters = {'evalue': Float,
              'maxaccepts': Int % Range(1, None),
              'perc_identity': Float % Range(0.0, 1.0, inclusive_end=True),
              'query_cov': Float % Range(0.0, 1.0, inclusive_end=True),
              'strand': Str % Choices(['both', 'plus', 'minus']),
              'output_no_hits': Bool,
              }

parameter_descriptions = {
    'evalue': 'BLAST expectation value (E) threshold for saving hits.',
    'strand': ('Align against reference sequences in forward ("plus"), '
               'reverse ("minus"), or both directions ("both").'),
    'maxaccepts': ('Maximum number of hits to keep for each query. BLAST will '
                   'choose the first N hits in the reference database that '
                   'exceed perc_identity similarity to query. NOTE: the '
                   'database is not sorted by similarity to query, so these '
                   'are the first N hits that pass the threshold, not '
                   'necessarily the top N hits.'),
    'perc_identity': ('Reject match if percent identity to query is lower.'),
    'query_cov': 'Reject match if query alignment coverage per high-'
                 'scoring pair is lower. Note: this uses blastn\'s '
                 'qcov_hsp_perc parameter, and may not behave identically '
                 'to the query_cov parameter used by classify-consensus-'
                 'vsearch.',
    'output_no_hits': 'Report both matching and non-matching queries. '
                      'WARNING: always use the default setting for this '
                      'option unless if you know what you are doing! If '
                      'you set this option to False, your sequences and '
                      'feature table will need to be filtered to exclude '
                      'unclassified sequences, otherwise you may run into '
                      'errors downstream from missing feature IDs. Set to '
                      'FALSE to mirror default BLAST search.',
}

blast6_output = ('search_results', FeatureData[BLAST6])

blast6_output_description = {'search_results': 'Top hits for each query.'}


# Note: name should be changed to blastn if we do NOT generalize this function
plugin.methods.register_function(
    function=blast,
    inputs=inputs,
    parameters=parameters,
    outputs=[blast6_output],
    input_descriptions=input_descriptions,
    parameter_descriptions=parameter_descriptions,
    output_descriptions=blast6_output_description,
    name='BLAST+ local alignment search.',
    description=('Search for top hits in a reference database via local '
                 'alignment between the query sequences and reference '
                 'database sequences using BLAST+. Returns a report '
                 'of the top M hits for each query (where M=maxaccepts).'),
    citations=[citations['camacho2009blast+']]
)


plugin.pipelines.register_function(
    function=classify_consensus_blast,
    inputs={**inputs,
            'reference_taxonomy': FeatureData[Taxonomy]},
    parameters={**parameters,
                **min_consensus_param,
                'unassignable_label': Str},
    outputs=[classification_output, blast6_output],
    input_descriptions={**input_descriptions,
                        'reference_taxonomy': 'reference taxonomy labels.'},
    parameter_descriptions={
        **parameter_descriptions,
        **min_consensus_param_description,
        'unassignable_label': 'Annotation given to sequences without any hits.'
    },
    output_descriptions={**classification_output_description,
                         **blast6_output_description},
    name='BLAST+ consensus taxonomy classifier',
    description=('Assign taxonomy to query sequences using BLAST+. Performs '
                 'BLAST+ local alignment between query and reference_reads, '
                 'then assigns consensus taxonomy to each query sequence from '
                 'among maxaccepts hits, min_consensus of which share '
                 'that taxonomic assignment. Note that maxaccepts selects the '
                 'first N hits with > perc_identity similarity to query, '
                 'not the top N matches. For top N hits, use '
                 'classify-consensus-vsearch.'),
    citations=[citations['camacho2009blast+']]
)
