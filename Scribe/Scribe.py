import pandas as pd
import numpy as np
from .causal_network import cmi

def causal_net_dynamics_coupling(adata, genes=None, guide_keys=None, t0_key='spliced', t1_key='velocity', normalize=True, copy=False):
    """Infer causal networks with dynamics-coupled single cells measurements.
    Network inference is a insanely challenging problem which has a long history and that none of the existing algorithms work well.
    However, it's quite possible that one or more of the algorithms could work if only they were given enough data. Single-cell
    RNA-seq is exciting because it provides a ton of data. Somewhat surprisingly, just having a lot of single-cell RNA-seq data
    won't make causal inference work well. We need a fundamentally better type of measurement that couples information across
    cells and across time points. Experimental improvements are coming now, and whether they are sufficient to power methods
    like Scribe is important future work. For example, the recent developed computational algorithm (La Manno et al. 2018) estimates
    the levels of new (unspliced) versus mature (spliced) transcripts from single-cell RNA-seq data for free. Moreover, exciting
    experimental approaches, like single cell SLAM-seq methods (Hendriks et al. 2018; Erhard et al. 2019; Cao, Zhou, et al. 2019)
    are recently developed that measures the transcriptome of two time points of the same cells. Datasets generated from those methods
    will provide improvements of causal network inference as we comprehensively demonstrated from the manuscript. This function take
    advantages of those datasets to infer the causal networks.

    We note that those technological advance may be still not sufficient, radically different methods, for example something like
    highly multiplexed live imaging that can record many genes may be needed.

    Arguments
    ---------
    adata: `anndata`
        Annotated data matrix.
    genes: `List`

    guide_keys: `List` (default: None)
        Whether to scale the expression or velocity values into 0 to 1 before calculating causal networks.
    t0_key: `str` (default: spliced)
        Key corresponds to the transcriptome of the initial time point, for example spliced RNAs from RNA velocity, old RNA
        from scSLAM-seq data.
    t1_key: `str` (default: velocity)
        Key corresponds to the transcriptome of the next time point, for example unspliced RNAs (or estimated velocitym,
        see Fig 6 of the Scribe preprint) from RNA velocity, old RNA from scSLAM-seq data.
    normalize: `bool`
        Whether to scale the expression or velocity values into 0 to 1 before calculating causal networks.
    copy: `bool`
        Whether to return a copy of the adata or just update adata in place. 

    Returns
    ---------
    An update AnnData object with inferred causal network stored as a matrix related to the key `causal_net` in the `uns` slot.
    """

    if genes is None and guide_keys is None:
        genes = adata.var_names.values.tolist()

    if guide_keys is not None:
        genes = np.unique(adata.obs[guide_keys].tolist())
        genes = np.setdiff1d(genes, ['*', 'nan', 'neg'])

        idx_var = [vn in genes for vn in adata.var_names]
        idx_var = np.argwhere(idx_var)
        genes = adata.var_names.values[idx_var.flatten()].tolist() #[idx_var]

    # support sparse matrix:
    tmp = pd.DataFrame(adata.layers[t0_key].todense())
    tmp.index = adata.obs_names
    tmp.columns = adata.var_names
    spliced = tmp.loc[:, genes]
    tmp = pd.DataFrame(adata.layers[t1_key]) #
    tmp.index = adata.obs_names
    tmp.columns = adata.var_names
    velocity = tmp.loc[:, genes]
    velocity[pd.isna(velocity)] = 0 # set NaN value to 0

    if normalize:
        spliced = (spliced - spliced.mean()) / (spliced.max() - spliced.min())
        velocity = (velocity - velocity.mean()) / (velocity.max() - velocity.min())

    causal_net = pd.DataFrame({node_id: [np.nan for i in genes] for node_id in genes}, index=genes)

    for g_a in genes:
        for g_b in genes:
            if g_a == g_b:
                continue
            else:
                x_orig = spliced.loc[:, g_a].tolist()
                y_orig = (spliced.loc[:, g_b] + velocity.loc[:, g_b]).tolist()
                z_orig = velocity.loc[:, g_b].tolist()

                # input to cmi is a list of list
                x_orig = [[i] for i in x_orig]
                y_orig = [[i] for i in y_orig]
                z_orig = [[i] for i in z_orig]
                causal_net.loc[g_a, g_b] = cmi(x_orig, y_orig, z_orig)

    adata.uns['causal_net'] = causal_net

#     logg.info('     done', time = True, end = ' ' if settings.verbosity > 2 else '\n')
#     logg.hint('perturbation_causal_net \n'
#               '     matrix is added to adata.uns')

    return adata if copy else None

