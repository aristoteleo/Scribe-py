import dynamo as dyn
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import isspmatrix

adata = dyn.read('/Users/xqiu/Dropbox/Projects/dynamo-release/debug/data/adata.h5ad')
gene_list = ['Neurog3', 'Pax6', 'Pdx1', 'Pax4', 'Arx', 'Actb', 'Rplp0', 'Gapdh']

import Scribe as sc

from Scribe.Scribe import causal_net_dynamics_coupling as Scribe

Scribe(adata[:, gene_list], t0_key='spliced', t1_key='unspliced')

sc.pl.viz_response(adata, np.array([['Pdx1', 'Pax4']]), grid_num=25, log=True) #

