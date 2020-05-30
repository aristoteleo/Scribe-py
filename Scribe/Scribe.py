from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.sparse import isspmatrix

from .causal_network import cmi
CLR_DDOF = 1


def causal_net_dynamics_coupling(adata,
                                 TFs=None,
                                 Targets=None,
                                 guide_keys=None,
                                 t0_key='spliced',
                                 t1_key='velocity',
                                 normalize=True,
                                 drop_zero_cells=False,
                                 copy=False):
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
    TFs: `List` or `None` (default: None)
        The list of transcription factors that will be used for casual network inference.
    Targets: `List` or `None` (default: None)
        The list of target genes that will be used for casual network inference.
    guide_keys: `List` (default: None)
        The key of the CRISPR-guides, stored as a column in the .obs attribute. This argument is useful
        for identifying the knockout or knockin genes for a perturb-seq experiment. Currently not used.
    t0_key: `str` (default: spliced)
        Key corresponds to the transcriptome of the initial time point, for example spliced RNAs from RNA velocity, old RNA
        from scSLAM-seq data.
    t1_key: `str` (default: velocity)
        Key corresponds to the transcriptome of the next time point, for example unspliced RNAs (or estimated velocitym,
        see Fig 6 of the Scribe preprint) from RNA velocity, old RNA from scSLAM-seq data.
    normalize: `bool`
        Whether to scale the expression or velocity values into 0 to 1 before calculating causal networks.
    drop_zero_cells: `bool` (Default: True)
        Whether to drop cells that with zero expression for either the potential regulator or potential target. This
        can signify the relationship between potential regulators and targets.
    copy: `bool`
        Whether to return a copy of the adata or just update adata in place.

    Returns
    ---------
        An update AnnData object with inferred causal network stored as a matrix related to the key `causal_net` in the `uns` slot.
    """

    if TFs is None:
        TFs = adata.var_names.tolist()
    else:
        TFs = adata.var_names.intersection(TFs).tolist()
        if len(TFs) == 0:
            raise Exception(f"The adata object has no gene names from .var_name that intersects with the TFs list you provided")

    if Targets is None:
        Targets = adata.var_names.tolist()
    else:
        Targets = adata.var_names.intersection(Targets).tolist()
        if len(Targets) == 0:
            raise Exception(f"The adata object has no gene names from .var_name that intersect with the Targets list you provided")

    if guide_keys is not None:
        guides = np.unique(adata.obs[guide_keys].tolist())
        guides = np.setdiff1d(guides, ['*', 'nan', 'neg'])

        idx_var = [vn in guides for vn in adata.var_names]
        idx_var = np.argwhere(idx_var)
        guides = adata.var_names.values[idx_var.flatten()].tolist()

    # support sparse matrix:
    genes = TFs + Targets
    genes = np.unique(genes)
    tmp = pd.DataFrame(adata[:, genes].layers[t0_key].todense()) if isspmatrix(adata.layers[t0_key]) \
        else pd.DataFrame(adata[:, genes].layers[t0_key])
    tmp.index = adata.obs_names
    tmp.columns = adata[:, genes].var_names
    spliced = tmp

    tmp = pd.DataFrame(adata[:, genes].layers[t1_key].todense()) if isspmatrix(adata.layers[t1_key]) \
        else pd.DataFrame(adata[:, genes].layers[t1_key])
    tmp.index = adata.obs_names
    tmp.columns = adata[:, genes].var_names
    velocity = tmp
    velocity[pd.isna(velocity)] = 0  # set NaN value to 0

    if normalize:
        spliced = (spliced - spliced.mean()) / (spliced.max() - spliced.min())
        velocity = (velocity - velocity.mean()) / (velocity.max() - velocity.min())

    causal_net = pd.DataFrame(columns=Targets, index=TFs)

    for g_a in tqdm(TFs, desc=f"Calculate causality score (RDI) from each TF to potential target:"):
        for g_b in Targets:
            if g_a == g_b:
                continue
            else:
                x_orig = spliced.loc[:, g_a]
                y_orig = (spliced.loc[:, g_b] + velocity.loc[:, g_b]) if t1_key is 'velocity' else velocity.loc[:, g_b]
                z_orig = spliced.loc[:, g_b]

                if drop_zero_cells:
                    xyz_orig = x_orig + y_orig + z_orig
                    x_orig, y_orig, z_orig = x_orig[xyz_orig > 0].tolist(), y_orig[xyz_orig > 0].tolist(), \
                                             z_orig[xyz_orig > 0].tolist()

                # input to cmi is a list of list
                x_orig = [[i] for i in x_orig]
                y_orig = [[i] for i in y_orig]
                z_orig = [[i] for i in z_orig]

                causal_net.loc[g_a, g_b] = cmi(x_orig, y_orig, z_orig)

    adata.uns['causal_net'] = {"RDI": causal_net.fillna(0)}

#     logg.info('     done', time = True, end = ' ' if settings.verbosity > 2 else '\n')
#     logg.hint('perturbation_causal_net \n'
#               '     matrix is added to adata.uns')

    return adata if copy else None

def CLR(causality_mat, zscore_both_dim=None):
    """Calculate the context likelihood of relatedness from mutual information. Note that the background mutual information or (RDI)
    uses the same data. code adapted from https://github.com/flatironinstitute/inferelator/blob/master/inferelator/regression/mi.py"""

    # if symmetric, then it is non-directional; otherwise it is directional
    zscore_both_dim = check_symmetric(causality_mat) if zscore_both_dim is None else zscore_both_dim
    mat = causality_mat.values if isinstance(causality_mat, pd.DataFrame) else causality_mat.A \
        if isspmatrix(causality_mat) else causality_mat

    # Calculate the zscore for rows
    z_row = np.round(mat, 10)  # Rounding so that float precision differences don't turn into huge CLR differences
    z_row = np.subtract(z_row, np.mean(mat, axis=1)) if mat.shape[0] == mat.shape[1] \
        else np.subtract(z_row, np.mean(mat, axis=1)[:, None])
    z_row = np.divide(z_row, np.std(mat, axis=1, ddof=CLR_DDOF)) if mat.shape[0] == mat.shape[1] \
        else np.divide(z_row, np.std(mat, axis=1, ddof=CLR_DDOF)[:, None])
    z_row[z_row < 0] = 0

    if zscore_both_dim:
        z_col = np.round(mat, 10)  # Rounding so that float precision differences don't turn into huge CLR differences
        z_col = np.subtract(z_col, np.mean(mat, axis=0)) if mat.shape[0] == mat.shape[1] \
            else np.subtract(z_col, np.mean(mat, axis=0)[None, :])
        z_col = np.divide(z_col, np.std(mat, axis=0, ddof=CLR_DDOF)) if mat.shape[0] == mat.shape[1] \
            else np.divide(z_col, np.std(mat, axis=0, ddof=CLR_DDOF)[None, :])

        z_col[z_col < 0] = 0

        res = np.sqrt((np.square(z_row) + np.square(z_col)) / 2)
    else:
        res = np.sqrt(np.square(z_row))

    if isinstance(causality_mat, pd.DataFrame):
        res = pd.DataFrame(res)
        res.index, res.columns = causality_mat.index, causality_mat.columns

    return res

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    if a.shape[0] != a.shape[1]:
        res = False
    else:
        if isinstance(a, pd.DataFrame):
            a = a.fillna(-1)
            a = a.values
        if isspmatrix(a):
            a.data = np.nan_to_num(a.data)
            res = (abs(a-a.T) > atol).nnz == 0
        else:
            res = np.allclose(a, a.T, rtol=rtol, atol=atol)

    return res
