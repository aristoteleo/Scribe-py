import numpy as np
import pandas as pd
import scipy
import numpy.matlib as matlib
import scipy.spatial as ss
import copy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from igraph import *
def sqdist (a,b):
    """calculate the square distance between a, b

    Arguments
    ---------
        a: 'np.ndarray'
            A matrix with :math:`D \times N` dimension
        b: 'np.ndarray'
            A matrix with :math:`D \times N` dimension

    Returns
    -------
    dist: 'np.ndarray'
        A numeric value for the different between a and b
    """
    # sum(a**2).tolist()
    # sum(b**2).tolist()
    # np.matrix(a).T * np.matrix(b)   #crossprod
    # rep_list = []
    # for i in range(len(b.shape[1])):
    #     rep_list = rep_list+[i for item in aa for i in item]
    # aa_repmat = np.array(rep_list).reshape(b.shape[1],len(aa)).T
    #
    # for i in range(len(a.shape[1])):
    #     rep_list = rep_list+[i for item in bb for i in item]
    # bb_repmat = np.array(rep_list).reshape(a.shape[1],len(bb))
    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = a.T.dot(b)

    aa_repmat = matlib.repmat(aa[:, None], 1, b.shape[1])
    bb_repmat = matlib.repmat(bb[None, :], a.shape[1], 1)

    dist = abs(aa_repmat + bb_repmat - 2 * ab)

    return dist

def repmat (X, m, n):
    """This function returns an array containing m (n) copies of A in the row (column) dimensions. The size of B is
    size(A)*n when A is a matrix.For example, repmat(np.matrix(1:4), 2, 3) returns a 4-by-6 matrix.

    Arguments
    ---------
        X: 'np.ndarray'
            An array like matrix.
        m: 'int'
            Number of copies on row dimension
        n: 'int'
            Number of copies on column dimension

    Returns
    -------
    xy_rep: 'np.ndarray'
        A matrix of repmat
    """
    # x_rep = X
    # for i in range(n-1):
    #     x_rep = np.hstack((x_rep,X))
    # xy_rep = x_rep
    # for i in range(m-1):
    #     xy_rep = np.vstack((xy_rep,x_rep))
    xy_rep = matlib.repmat(X, m, n)

    return xy_rep

def eye (m, n):
    """Equivalent of eye (matlab)

    Arguments
    ---------
        m: 'int'
            Number of rows
        n: 'int'
            Number of columns

    Returns
    -------
    mat: 'np.ndarray'
        A matrix of eye
    """
    # mat = np.zeros((n,m))
    # for i in range(min(len(mat), len(mat[0]))):
    #     mat[i][i] = 1

    mat = np.eye(m, n)
    return mat

#
# def sqdist(a, b):
#     """calculate the square distance between a, b
#
#     Arguments
#     ---------
#         a: 'np.ndarray'
#             Array like matrix
#         b: 'np.ndarray'
#         Array like matrix
#
#     Returns
#     -------
#     dist: 'np.ndarray'
#         The distance matrix
#     """
#     aa = sum(a ** 2).tolist()
#     bb = sum(b ** 2).tolist()
#     ab = np.matrix(a).T * np.matrix(b)  # crossprod
#     rep_list = []
#     for i in range(len(b.shape[1])):
#         rep_list = rep_list + [i for item in aa for i in item]
#     aa_repmat = np.array(rep_list).reshape(b.shape[1], len(aa)).T
#
#     for i in range(len(a.shape[1])):
#         rep_list = rep_list + [i for item in bb for i in item]
#     bb_repmat = np.array(rep_list).reshape(a.shape[1], len(bb))
#     dist = abs(aa_repmat + bb_repmat - 2 * ab)
#     return dist
#
# def repmat (X, m, n):
#     """This function returns an array containing m (n) copies of A in the row (column) dimensions. The size of B is
#     size(A)*n when A is a matrix.For example, repmat(np.matrix(1:4), 2, 3) returns a 4-by-6 matrix.
#
#     Arguments
#     ---------
#         X: 'np.ndarray'
#             An array like matrix.
#         m: 'int'
#             Number of copies on row dimension
#         n: 'int'
#             Number of copies on column dimension
#
#     Returns
#     -------
#     xy_rep: 'np.ndarray'
#         A matrix of repmat
#     """
#     x_rep = X
#     for i in range(n-1):
#         x_rep = np.hstack((x_rep,X))
#     xy_rep = x_rep
#     for i in range(m-1):
#         xy_rep = np.vstack((xy_rep,x_rep))
#
#     return xy_rep
#
# def eye (m, n):
#     """Equivalent of eye (matlab)
#
#     Arguments
#     ---------
#         m: 'int'
#             Number of rows
#         n: 'int'
#             Number of columns
#
#     Returns
#     -------
#     mat: 'np.ndarray'
#         A matrix of eye
#     """
#     mat = np.zeros((n,m))
#     for i in range(min(len(mat), len(mat[0]))):
#         mat[i][i] = 1
#     return mat


def get_adjacency_MSTree(x):
    '''Calculate the Minimum Spanning Tree'''

    done_vertices = [0]
    adj = csr_matrix(x.shape, dtype=np.float64)
    while len(done_vertices) < x.shape[0]:
        minNum = np.inf

        for i in done_vertices:
            for j in range(x.shape[0]):
                if not (j in done_vertices):
                    if j < i:
                        num = x[i][j]
                    elif j > i:
                        num = x[j][i]
                    else:
                        num = np.inf
                    if num < minNum:
                        minNum = num
                        index = j
                        if i > j:
                            index_i = i
                            index_j = j
                        else:
                            index_i = j
                            index_j = i
        adj[index_i, index_j] = minNum
        done_vertices = done_vertices + [index]
    return adj.toarray()


def principal_tree (X, MU = None, Lambda = 1.0, bandwidth = None, maxIter = 100, verbose = False):
    """Learn the principal tree from the same dimension of the noisy data.

    Arguments
    ---------
        X: 'np.narray'
            A matrix containing the original noisy data points
        MU: 'np.narray'
            A matrix
        Lambda: 'float' (Default: 1.0)
            XXXX
        bandwidth: 'float' (Default: None)
            Bandwidth
        maxIter: `int` (Default: 100)
            Maximal number of iteraction of principal tree algorithm
        verbose: `bool` (Default: False)
            A logic flag to determine whether or not running information should be returned.

    Returns
    -------
    MU,stree,sigma,Lambda,history: `np.ndarray`, `np.ndarray`, `float`, `float`, `np.narray`
        A tuple of four elements, including the final MU, stree, param.sigma, lambda and running history
    """
    D = X.shape[0]
    N = X.shape[1]
    old_obj = 0

    # initialize MU, lmabda
    if not MU:
        K = N
        MU = X
    else :
        K = MU.shape[1]

    Lambda = Lambda * N  # scale the parameter by N

    if not bandwidth:
        distsqX = sqdist(X, X)
        sigma = 0.01 * sum(sum(distsqX)) / (N ** 2)
    else:
        sigma = bandwidth

    history = pd.DataFrame(index=[i for i in range(100)], columns=['mu', 'stree', 'objs', 'mse', 'length'])
    for iter in range(maxIter):
        # Kruskal method to find a spanning tree
        distsMU = sqdist(MU, MU)
        # distsMU_low = np.tril(distsMU)
        # stree = get_adjacency_MSTree(distsMU_low)
        gp = Graph.Weighted_Adjacency(distsMU.tolist(), ADJ_LOWER)
        g_mst = gp.spanning_tree()
        stree = np.array(g_mst.get_adjacency(type = GET_ADJACENCY_LOWER, attribute='weight')._get_data())

        stree_ori = stree
        stree = stree+stree.T
        e = stree != 0

        history.iloc[iter]['mu'] = MU

        # update data assignment matrix
        distMUX = sqdist(MU, X)
        # t = np.array([])
        # for i in range(distMUX.shape[0]):
        #     t = np.append(t, min(distMUX[i]))
        # min_dist = repmat(t, K, 1)
        min_dist = repmat(np.min(distMUX, 1), K, 1)
        tmp_distMUX = distMUX - min_dist
        tmp_R = np.exp(-tmp_distMUX.T/ sigma)
        R = tmp_R / repmat(sum(tmp_R.T).reshape(-1, 1), 1, K)

        # compute objective function
        obj1 = - sigma * sum(np.log(sum(np.exp(- tmp_distMUX / sigma))) - min_dist[0, :] / sigma)
        reg = sum(sum(stree))  # full
        obj = (obj1 + 0.5 * Lambda * reg) / N
        history.iloc[iter]['obj'] = obj

        # t = np.array([])
        # dist_tmp = distMUX.T
        # for i in range(dist_tmp.shape[0]):
        #     t = np.append(t, min(dist_tmp[i]))
        # projd = t
        projd = np.min(distMUX, 0)
        mse = np.mean(projd)

        history.iloc[iter]['mse'] = mse

        # length of the structure
        history.iloc[iter]['length'] = reg

        # terminate condition
        if verbose:
            print('iter=', iter, ', obj=', obj, ', mse=', mse, ', len=', reg)
        if iter > 0:
            if (abs((obj - old_obj) / old_obj) < 1e-3):
                break

        L = np.diag(sum(e)) - np.array(e, dtype=int)
        MU = X.dot(R).dot(scipy.linalg.inv(Lambda * L + np.diag(sum(R))))

        old_obj = obj

    return MU,stree,sigma,Lambda,history

import scipy.io
tmp = scipy.io.loadmat('/Users/xqiu/Downloads/PAMI2015-code/toy/tree_300.mat')
X = tmp['X']

Lambda = 1
bandwidth = 0.0050
MU, stree, sigma, Lambda, history = principal_tree(X.T, MU = None, Lambda = Lambda, bandwidth = bandwidth, maxIter = 100, verbose = False);
