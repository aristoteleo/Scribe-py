.. Scribe documentation master file, created by
   sphinx-quickstart on Sat Mar 16 23:24:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Scribe's documentation!
==================================

**Scribe** is a toolkit for analyzing gene regulation network. It can be used to calculate **RDI**, **cRDI**, **uRDI**, **ucRDI**.

First, you need to import Scribe
as::
    import Scribe as scr

And then you need to read data
use::
    from scr.read_export import load_anndata,read
    adata = read(filename)
    model=load_anndata(adata)
Data is a matrix. The rows is an index of GINE_ID with RUN_ID(Multiple index) and columns is cell number

Before we calculate the gene causal network we must initialize
roc::
    ccm_roc = {}
    rdi_roc = {}
    crdi_roc = {}
    urdi_roc = {}
    ucrdi_roc = {}

Now we can calculate RDI(or uRDI) matrix and compare with real networks,
use::
    true_graph_path = trueNetFilePath
    model.rdi(delays=[1,2,3], number_of_processes=1, uniformization=False, differential_mode=False)
    rdi_auc, rdi_roc["x"], rdi_roc["y"] = model.roc(results=model.rdi_results["MAX"], true_graph_path=true_graph_path)

If you want to calculate uRDI(or ucRDI) just
set::
    uniformization = True

Computing CRDI is similar to the above process. Use **model.crdi()** just like rdi.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


