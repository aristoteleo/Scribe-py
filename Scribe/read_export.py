'''
Function to take data from anndata and append the data to expression_raw, etc. for running Scribe 
AnnData object with n_obs × n_vars = 640 × 11 
    obs: 'cell_type', 'dpt_pseudotime', 'dpt_groups', 'dpt_order', 'dpt_order_indices'
    uns: 'cell_type_colors', 'diffmap_evals', 'dpt_changepoints', 'dpt_groups_colors', 'dpt_grouptips', 'draw_graph', 'highlights', 'iroot', 'neighbors', 'tmax_write'
    obsm: 'X_tsne', 'X_draw_graph_fr', 'X_diffmap'
''' 
from . import causal_network            #################for ipython
import pandas as pd
import numpy as np

def load_anndata(anndata,is_scale=False,keys='dpt_groups'): # need to support multi-run next
    model =causal_network.causal_model()
    if is_scale == False :
        expression_raw = anndata.X
    else:
        expression_raw = anndata.X._scale

    df = pd.DataFrame(expression_raw.transpose(), index = anndata.var_names.values.tolist())
    df.index.name = 'GENE_ID'



    if keys=='dpt_groups':
        keys=anndata.obs.dpt_groups.transpose().values.tolist()
        uniq_keys = anndata.obs.dpt_groups.unique()
        uniq_keys = sorted(uniq_keys)
    ############### i can add other obs_key here

    for key in uniq_keys :

        if key == '1':####################################################
            continue
        cur_id = np.where(np.array(keys) == str(key))

        sort_cur_id = np.argsort(anndata.obs.dpt_pseudotime.iloc[cur_id].values)


        cur_expression_mat = df[df.columns[cur_id]] # Remember, Python is 0-offset! The "3rd" entry is at slot 2.
        cur_expression_mat = cur_expression_mat[cur_expression_mat.columns[sort_cur_id]]

        runid=np.array(df.index)
        i=0
        while i<len(df.index) :
            runid[i]=key
            i=i+1
        arrays=[df.index,runid]
        index_col=pd.MultiIndex.from_arrays(arrays, names=('GINE_ID','RUN_ID'))
        if key=='0':
            expression_mat_=pd.DataFrame(cur_expression_mat.values.tolist(),index=index_col)
        else :
            cur_expression_mat_=pd.DataFrame(cur_expression_mat.values.tolist(),index=index_col)
            expression_mat_=pd.concat([expression_mat_,cur_expression_mat_],axis=0)

    model.expression_raw = expression_mat_
    model.expression = expression_mat_
    model.velocity = expression_mat_
    model.node_ids = expression_mat_.index.levels[0]
    model.run_ids = expression_mat_.index.levels[1]
    return model
    
# read loom, etc. 
