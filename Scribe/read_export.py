'''
Function to take data from anndata and append the data to expression_raw, etc. for running Scribe
AnnData object with n_obs × n_vars = 640 × 11
    obs: 'cell_type', 'dpt_pseudotime', 'dpt_groups', 'dpt_order', 'dpt_order_indices'
    uns: 'cell_type_colors', 'diffmap_evals', 'dpt_changepoints', 'dpt_groups_colors', 'dpt_grouptips', 'draw_graph', 'highlights', 'iroot', 'neighbors', 'tmax_write'
    obsm: 'X_tsne', 'X_draw_graph_fr', 'X_diffmap'
'''
from . import causal_network
from . import logging as logg

import pandas as pd
import numpy as np

# append velocity data, etc. 
def load_anndata(anndata, is_scale=False):
    """Convert an anndata object to a causal_model object.

    Arguments
    ---------
    anndata: `anndata`
        Annotated data matrix.
    keys: `str`
        Column in obs used to set run id.
    is_scale: `bool`
        If value is true, read anndata.X._scale, else anndata.X.

    Returns
    ---------
    return: causal_network object.
    """

    model =causal_network.causal_model()
    logg.info('Create causal_model successfully')

    order_exprs_mat = True
    if is_scale == False :
        expression_raw = anndata.X
    else:
        expression_raw = anndata.X._scale
    df = pd.DataFrame(expression_raw.transpose(), index = anndata.var_names.tolist())
    df.index.name = 'GENE_ID'

    if "dpt_groups" in anndata.obs_keys(): # if pseudotime and branch of the dataset is assigned by scanpy, use the following 
        branch_key, pseudotime_key = 'dpt_groups', 'dpt_pseudotime'
    elif "Branch" in anndata.obs_keys():  # if pseudotime and branch of the dataset is assigned by Monocle, use the following
        branch_key, pseudotime_key = 'Branch', 'Pseudotime' 
    else:
        order_exprs_mat = False

    if order_exprs_mat:
        keys=anndata.obs[branch_key].tolist()
        uniq_keys = anndata.obs[branch_key].unique()
        uniq_keys = sorted(uniq_keys)

        for key in uniq_keys:
            if np.sum(anndata.obs[branch_key] == key) == 1:
                continue
            cur_id = np.where(np.array(keys) == str(key))

            sort_cur_id = np.argsort(anndata.obs[pseudotime_key].iloc[cur_id].values)

            cur_expression_mat, cur_obs_name = df[df.columns[cur_id]], anndata.obs_names[cur_id] # Remember, Python is 0-offset! The "3rd" entry is at slot 2.
            cur_expression_mat, cur_obs_name = cur_expression_mat[cur_expression_mat.columns[sort_cur_id]], cur_obs_name[sort_cur_id]

            runid=np.array(df.index)
            i=0

            runid[:] = key

            arrays=[df.index,runid]
            index_col=pd.MultiIndex.from_arrays(arrays, names=('GINE_ID','RUN_ID'))
            if key==uniq_keys[0]:
                expression_mat_=pd.DataFrame(cur_expression_mat.values.tolist(),index=index_col) # , columns=cur_obs_name
            else :
                cur_expression_mat_=pd.DataFrame(cur_expression_mat.values.tolist(),index=index_col) # , columns=cur_obs_name
                expression_mat_=pd.concat([expression_mat_,cur_expression_mat_],axis=0)
    else:
        expression_mat_ = expression_raw

    model.X = pd.DataFrame(np.transpose(anndata.X), index=anndata.var_names, columns=anndata.obs_names)
    model.expression_raw = expression_mat_
    model.expression = expression_mat_
    model.unspliced = anndata.layers['unspliced'] if 'unspliced' in anndata.layers.keys() else None
    model.spliced = anndata.layers['spliced'] if 'spliced' in anndata.layers.keys() else None
    model.velocity = anndata.layers['velocity'] if 'velocity' in anndata.layers.keys() else None

    if order_exprs_mat:
        model.node_ids = expression_mat_.index.levels[0]
        model.run_ids = expression_mat_.index.levels[1]
    else:
        model.node_ids = anndata.var_names.to_list()
        model.run_ids = [0]


    return model
    
# # read loom, etc.
# import Scribe as sc
# import scanpy as sp
#
# import numpy as np
#
# krumsiek11 = sp.read_h5ad('/Users/xqiu/Desktop/krumsiek11_blobs.h5ad')
# krumsiek11.obs['dpt_groups'] = krumsiek11.obs['clusters']
#
# adata = load_anndata(krumsiek11)
# # sc.gene_interaction_visualization.plot_lagged_drevi(krumsiek11,np.array([['Pu.1','Gata1'], ['Gata1','Fog1'], ['Gata2','Gata1']]),grid_num=25) #
