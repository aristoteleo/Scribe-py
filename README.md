## **Scribe**: Towards inferring causal gene regulatory networks from single cell expression Measurements
![Dynamo](https://pbs.twimg.com/media/DoLDC2nVsAAei7r?format=jpg&name=medium)

Single-cell transcriptome sequencing now routinely samples thousands of cells, potentially providing enough data to reconstruct *causal gene regulatory networks* from observational data. Here, we developed **Scribe**, a toolkit for detecting and visualizing causal regulations, and explore the potential for single-cell experiments to power network reconstruction. **Scribe** employs *Restricted Directed Information* to determine causality by estimating the strength of information transferred from a potential regulator to its downstream target by taking advantage of time-delays. We apply **Scribe** and other leading approaches for network reconstruction to several types of single-cell measurements and show that there is a dramatic drop in performance for "pseudotime” ordered single-cell data compared to live imaging data. We demonstrate that performing causal inference requires temporal coupling between measurements. We show that methods such as “*RNA velocity*” restore some degree of coupling through an analysis of chromaffin cell fate commitment. These analyses therefore highlight an important shortcoming in experimental and computational methods for analyzing gene regulation at single-cell resolution and point the way towards overcoming it.

## Installation

Note that this is our first alpha version of **Scribe** (as of Aug. 11th, 2019) python package. Scribe is still under active development. Stable version of Scribe will be released when it is ready. Until then, please use **Scribe** with caution. We welcome any bugs reports (via GitHub issue reporter) and especially code contribution  (via GitHub pull requests) of **Scribe** from users to make it an accessible, useful and extendable tool. For discussion about different usage cases, comments or suggestions related to our manuscript and questions regarding the underlying mathematical formulation of dynamo, we provided a google group [goolge group](https://groups.google.com/forum/#!forum/Scribe-user/). Scribe developers can be reached by <xqiu.sc@gmail.com>. To install the newest version of dynamo, you can git clone our repo and then use::

```sh
pip install directory_to_Scribe_py_repo/
```

Alternatively, You can install **Scribe** from source, using the following script:
```sh
pip install git+https://github.com:aristoteleo/Scribe-py
```

## Citation
Xiaojie Qiu, Arman Rahimzamani, Li Wang, Qi Mao, Timothy Durham, Jose L McFaline-Figueroa, Lauren Saunders, Cole Trapnell, Sreeram Kannan (2018): Towards inferring causal gene regulatory networks from single cell expression measurements. BioRxiv

biorxiv link: https://www.biorxiv.org/content/early/2018/09/25/426981

twitter link: https://twitter.com/coletrapnell/status/1044986820520435712

## Contribution 
If you want to contribute to the development of dynamo, please check out CONTRIBUTION instruction: [Contribution](https://github.com/aristoteleo/Scribe-py/blob/master/CONTRIBUTING.md)

## Documentation  
The documentation of dynamo package is available at [readthedocs](https://Scribe-py.readthedocs.io/en/latest/)
