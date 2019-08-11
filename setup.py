from setuptools import setup, find_packages
import numpy as np
from get_version import get_version 

setup(
    name="Scribe",
    __version__ = get_version(__file__),
    install_requires=['cvxopt>=1.2.3', 'pandas>=0.23.0', 'numpy>=1.14', 'scipy>=1.0', 'scikit-learn>=0.19.1', 
                      'pyccm>=0.4', 'statsmodels>=0.9.0', 'scanpy>=1.3.3', 'anndata>=0.6.18', 'loompy>=2.0.12', 
                      'matplotlib>=2.2', 'setuptools'],
    packages=find_packages(),
    include_dirs=[np.get_include()],
    author="Xiaojie Qiu,Arman Rahimzamani,Bingcheng Ren",
    author_email="xqiu.sc@gmail.com",
    description='Detect causality from single cell measurements',
    license='BSD',
    url="https://github.com/Xiaojieqiu/velocity_slam_seq",
    download_url=f"https://github.com/Xiaojieqiu/velocity_slam_seq",
    keywords=["RNAseq", "singlecell", "network", "causality", "velocity"]
    )
