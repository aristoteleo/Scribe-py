import scipy.spatial as ss
from scipy.special import digamma,gamma
from sklearn.neighbors import KernelDensity
from math import log,pi,exp
import numpy.random as nr
import numpy as np
# from cvxopt import matrix,solvers
# from random import gauss, uniform, randint, expovariate
from matplotlib import pyplot as plt
# from misc import rank_order
from copy import deepcopy
# from scipy.stats import rankdata


def vd(d):
    """Calculates the volume of a d-dimensional unit ball for Euclidean norm.

    `vd` takes a integer of dimensions and then calculate the volume of a d-dimensional unit ball for Euclidean norm
    using the formula: :math:`0.5 * d * log(pi) - log(gamma(0.5 * d + 1))`.

    Arguments
    ---------
        d: 'int'
            Number of dimension.

    Returns
    -------
    A numeric value for the d-dimensional unit ball for Euclidean norm
    """
    return 0.5 * d * log(pi) - log(gamma(0.5 * d + 1))


def entropy(x, k=5):
    """Estimates the entropy of a continuous random variable.

    `entropy` takes a continuous random variable and then estimates entropy using the KSG estimator. It relies on the
    cKDTree function in scipy to query the kNN with KDTree algorithm.

    Arguments
    ---------
        x: 'np.ndarray'
            Data matrix used for calculating the entropy.
        k: Number for nearest neighbors used in entropy calculation

    Returns
    -------
    A numeric value of entropy estimate
    """
    N = len(x)  # The number of observed samples
    # k = int(np.floor(np.sqrt(N)))
    d = len(x[0])  # The number of the dimensions of the data
    tree = ss.cKDTree(x)  # kd-tree for quick nearest-neighbor lookup
    knn_dis = [tree.query(point, k + 1, p=np.inf)[0][k] for point in
               x]  # distance to the kth nearest neighbor for all points
    ans = -digamma(k) + digamma(N)
    return ans + d * np.mean(map(log, knn_dis))


def mi(x_orig, y_orig, use_rank_order=False, k=5):
    """Estimates the mutual information of two random variables based on their observed values.

    `mi` takes two random variables :math:`x` and :math:`y` to estimate the mutual information between them using the KSG estimator.
    It relies on the cKDTree function in scipy to query the kNN with KDTree algorithm.

    Arguments
    ---------
        x_orig: `List`
            One random variable from the time-series data.
        y_orig: `List`
            Another random variable from the time-series data.
        use_rank_order: `bool` (default: False)
            Whether to use rank order instead of actual value for MI calculation.
        k: (Default: 5)
            Number for nearest neighbors used in entropy calculation

    Returns
    -------
    A numeric value of mutual information estimate
    """
    x = deepcopy(x_orig)
    y = deepcopy(y_orig)

    assert len(x) == len(y), "Lists should have same length"
    N = len(x)

    dx = len(x[0])
    dy = len(y[0])

    # if use_rank_order:
    #    x = rank_order(x)
    #    y = rank_order(y)

    data = np.concatenate((x, y), axis=1)


    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    # knn_dis = [tree_xy.query(point, k + 1, p=np.inf)[0][k] for point in data]
    knn_dis = tree_xy.query(data, k + 1, p=np.inf)[0]
    information_samples = [ digamma(N) for i in range(N)]

    for i in range(N):
        information_samples[i] += digamma(len(tree_xy.query_ball_point(data[i], knn_dis[i][k], p=np.inf )) -1)
        information_samples[i] += -digamma(len(tree_x.query_ball_point(x[i], knn_dis[i][k], p=np.inf )) - 1)
        information_samples[i] += -digamma(len(tree_y.query_ball_point(y[i], knn_dis[i][k], p=np.inf )) - 1)

    return np.mean(information_samples)

def cmi(x_orig, y_orig, z_orig, normalization=False, k=5):
    """Estimates the CONDITIONAL mutual information of :math:`x` and :math:`y` given :math:`z`.

    `cmi` takes two random variable :math:`x` and :math:`y` and estimated their mutual information conditioned on the
    third random variable :math:`z` using the KSG estimator. It relies on the cKDTree function in scipy to query the kNN
    with KDTree algorithm.

    Arguments
    ---------
        x_orig: `List`
            One random variable from the time-series data.
        y_orig: `List`
            Another random variable from the time-series data.
        z_orig: `List`
            Condition random variable for variables (:math:`x, y`) from the time-series data.
        use_rank_order: `bool` (default: False)
            Whether to use rank order instead of actual value for MI calculation.
        k: `int` (Default: 5)
            Number for nearest neighbors used in entropy calculation

    Returns
    -------
    A numeric value of conditional mutual information estimate
    """
    x = deepcopy(x_orig)
    y = deepcopy(y_orig)
    z = deepcopy(z_orig)

    #print(z_orig)
    #print('##########################')
    assert len(x) == len(y), "Lists should have same length"
    assert len(x) == len(z), "Lists should have same length"

    N = len(x)

    dx = len(x[0])
    dy = len(y[0])
    dz = len(z[0])

    if normalization:
       x /= np.std(x)
       y /= np.std(y)
       z /= np.std(z)

    data_xyz = np.concatenate((x, y, z), axis=1)
    data_xz = np.concatenate((x, z), axis=1)
    data_yz = np.concatenate((y, z), axis=1)


    tree_xyz = ss.cKDTree(data_xyz)
    tree_xz = ss.cKDTree(data_xz)
    tree_yz = ss.cKDTree(data_yz)
    tree_z = ss.cKDTree(z)

    knn_dis = [tree_xyz.query(point, k + 1, p=np.inf)[0][k] for point in data_xyz]
    # knn_dis = tree_xyz.query(data_xyz, k + 1, p=np.inf)[0][k]
    information_samples = [0 for i in range(N)]
    for i in range(N):
        information_samples[i] += digamma(len(tree_xyz.query_ball_point(data_xyz[i], knn_dis[i], p=np.inf)) -1)
        information_samples[i] += -digamma(len(tree_xz.query_ball_point(data_xz[i], knn_dis[i], p=np.inf)) - 1)
        information_samples[i] += -digamma(len(tree_yz.query_ball_point(data_yz[i], knn_dis[i], p=np.inf)) - 1)
        information_samples[i] += digamma(len(tree_z.query_ball_point(z[i], knn_dis[i], p=np.inf)) -1)
    return np.mean(information_samples)


def di_single_run(x,y, n=10, bagging=None):
    """Simulates the DIRECTED mutual information from :math:`x` to :math:`y` when you have a SINGLE run of the processes.

    `cmi` takes two random variable :math:`x` and :math:`y` and estimated their mutual information conditioned on the
    third random variable :math:`z` using the KSG estimator. It relies on the cKDTree function in scipy to query the kNN
    with KDTree algorithm.

    Arguments
    ---------
        x: `List`
            One random variable from the time-series data.
        y: `List`
            Another random variable from the time-series data.
        n: `int`
            Determines the the number previous time samples upon which the mi is conditioned.
        bagging: `bool` (default: None)
            This argument is not used anymore.

    Returns
    -------
    A numeric value of entropy estimate
    """
    assert len(x[0]) == len(y[0]), "The number of time samples has to be the same for X and Y"
    tau = n
    tot_len = len(x) - tau
    x_past = x[tau-1:tau-1+tot_len]
    y_past = y[tau-1:tau-1+tot_len]
    for i in range(2,n+1):
        x_past = np.concatenate( (x[tau-i:tau-i+tot_len], x_past), axis=1)
        y_past = np.concatenate( (y[tau-i:tau-i+tot_len], y_past), axis=1)
    return cmi(x_past, y[tau:tau+tot_len], y_past)


def umi(x, y, k=5, density_estimation_method="kde", k_density=5, bw=.01):
    """Calculates the uniformed mutual information where the distribution for :math:`x` is replaced by a uniform distribution.

    `umi` takes two random variable x and y and estimated their mutual using the KSG estimator while x is replaced by a
    uniform distribution.

    Arguments
    ---------
        x: `List`
            One random variable from the time-series data.
        y: `List`
            Another random variable from the time-series data.
        k: `int` (Default: 5)
            Number for nearest neighbors used in entropy calculation
        density_estimation_method: `str` (Default: `kde`)
            Which 2D density estimator you would like to use. `kde` is kde estimator while `knn` is knn based estimator.
        k_density: `bool` (default: False)
            The number of k nearest neighbors you would like to use when calculating the density (only applicable when
            density_estimation_method is to be `knn` or using knn based density estimation).
        bw: `float` (default: 0.1)
            Bindwidth used for the kernel density estimator.

    Returns
    -------
    A estimated uniform mutual information value between two variables (x, y) where the distribution for the x is replaced
    by a uniform distribution.
    """
    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    N = len(x)
    dx = len(x[0])
    dy = len(y[0])
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    if density_estimation_method.lower()=="kde":
        kernel = KernelDensity(bandwidth=bw)
        kernel.fit(x)
        kde = np.exp(kernel.score_samples(x))
        weight = (1 / kde) / np.mean(1 / kde)

    elif density_estimation_method.lower()=="knn":
        knn_dis = [tree_x.query(point, k_density + 1, p=np.inf)[0][k_density] for point in x]
        density_estimate = np.array([float(k_density) / N / knn_dis[i] ** dx for i in range(len(knn_dis))])
        weight = (1 / density_estimate) / np.mean(1 / density_estimate)

    else:
        raise ValueError("The density estimation method is not recognized")

    

    knn_dis = [tree_xy.query(point, k + 1, p=2)[0][k] for point in data]
    ans = digamma(k) + 2 * log(N - 1) - digamma(N) + vd(dx) + vd(dy) - vd(dx+dy)

    weight_y = np.zeros(N)
    for i in range(N):
        weight_y[i] = np.sum(weight[j] for j in tree_y.query_ball_point(y[i], knn_dis[i], p=2)) - weight[i]
    weight_y *= N/np.sum(weight_y)

    for i in range(N):
        nx = len(tree_x.query_ball_point(x[i], knn_dis[i], p=2)) - 1
        ny = np.sum(weight[j] for j in tree_y.query_ball_point(y[i], knn_dis[i], p=2)) - weight[i]
        ans += -weight[i] * log(nx) / N
        # ans += -ny * log(ny) / N / (len(tree_y.query_ball_point(y[i], knn_dis[i], p=2))-1)
        ans += -weight[i] * log(ny) / N
    return ans


def alternate_umi(x, y, k=5, density_estimation_method="kde", k_density=5, bw=.2):
    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    N = len(x)
    dx = len(x[0])
    dy = len(y[0])
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    if density_estimation_method.lower()=="kde":
        kernel = KernelDensity(bandwidth=bw)
        kernel.fit(x)
        kde = np.exp(kernel.score_samples(x))
        weight = (1 / kde) / np.mean(1 / kde)

    elif density_estimation_method.lower()=="knn":
        knn_dis = [tree_x.query(point, k_density + 1, p=np.inf)[0][k_density] for point in x]
        density_estimate = np.array([float(k_density) / N / knn_dis[i] ** dx for i in range(len(knn_dis))])
        weight = (1 / density_estimate) / np.mean(1 / density_estimate)

    else:
        raise ValueError("The density estimation method is not recognized")

    knn_dis = [tree_xy.query(point, k + 1, p=2)[0][k] for point in data]
    ans = log(k) + 2 * log(N - 1) - digamma(N) + vd(dx) + vd(dy) - vd(dx+dy)

    # weight_y = np.zeros(N)
    # for i in range(N):
    #     weight_y[i] = np.sum(weight[j] for j in tree_y.query_ball_point(y[i], knn_dis[i], p=2)) - weight[i]
    # weight_y *= N/np.sum(weight_y)

    for i in range(N):
        nx = len(tree_x.query_ball_point(x[i], knn_dis[i], p=2)) - 1
        ny = len(tree_y.query_ball_point(y[i], knn_dis[i], p=2)) - 1
        ans += -weight[i] * log(nx) / N
        ans += -weight[i] * log(ny) / N
        # for j in tree_y.query_ball_point(y[i], knn_dis[i], p=2):
            # ans += -weight[j] * log(weight[j]) /N/ny
        # ans += -weight[i] * log(weight[i]) / N

    return ans


def cumi(x_orig, y_orig, z_orig, normalization=False, k=5, density_estimation_method="kde", k_density=5, bw=.01):
    """Calculates the uniformed conditional mutual information where the distribution for :math:`x` and :math:`z` is replaced by a uniform distribution.

    `cumi` takes two random variable :math:`x` and :math:`y` and estimated their mutual information conditioned on the
    third random variable :math:`z` using the KSG estimator while :math:`x`, :math:`y` is replaced by a uniform distribution.

    Arguments
    ---------
        x_orig: `List`
            One random variable from the time-series data.
        y_orig: `List`
            Another random variable from the time-series data.
        z_orig: `List`
            Another random variable from the time-series data.
        normalization: `bool` (Default: False)
            Whether to normalize the expression of :math:`x, y, z` by their standard deviation.
        k: `int` (Default: 5)
            Number for nearest neighbors used in entropy calculation
        density_estimation_method: `str` (Default: `kde`)
            Which 2D density estimator you would like to use. `kde` is kde estimator while `knn` is knn based estimator.
        k_density: `bool` (default: False)
            The number of k nearest neighbors you would like to use when calculating the density (only applicable when
            density_estimation_method is to be `knn` or using knn based density estimation).
        bw: `float` (default: 0.01)
            Bindwidth used for the kernel density estimator.

    Returns
    -------
    A estimated conditional mutual information value between two variables (x, y), conditioning on a third variable z where
    the distribution for the x, z is replaced by a uniform distribution.
    """
    x = deepcopy(x_orig)
    y = deepcopy(y_orig)
    z = deepcopy(z_orig)

    assert len(x) == len(y), "Lists should have same length"
    assert len(x) == len(z), "Lists should have same length"

    N = len(x)

    dx = len(x[0])
    dy = len(y[0])
    dz = len(z[0])

    if normalization:
       x /= np.std(x)
       y /= np.std(y)
       z /= np.std(z)

    data_xyz = np.concatenate((x, y, z), axis=1)
    data_xz = np.concatenate((x, z), axis=1)
    data_yz = np.concatenate((y, z), axis=1)

    tree_xyz = ss.cKDTree(data_xyz)
    tree_xz = ss.cKDTree(data_xz)
    tree_yz = ss.cKDTree(data_yz)
    tree_z = ss.cKDTree(z)

    if density_estimation_method.lower()=="kde":
        kernel = KernelDensity(bandwidth=bw)
        kernel.fit(data_xz)
        kde = np.exp(kernel.score_samples(data_xz))
        weight = (1 / kde) / np.mean(1 / kde)
    elif density_estimation_method.lower()=="knn":
        knn_dis = [tree_xz.query(point, k_density + 1, p=np.inf)[0][k_density] for point in data_xz]
        density_estimate = np.array([float(k_density) / N / knn_dis[i] ** (dx+dz) for i in range(len(knn_dis))])
        weight = (1 / density_estimate) / np.mean(1 / density_estimate)
    else:
        raise ValueError("The density estimation method is not recognized")

    knn_dis = [tree_xyz.query(point, k + 1, p=np.inf )[0][k] for point in data_xyz]
    information_samples = [ 0 for i in range(N)]
    for i in range(N):
        information_samples[i] += weight[i]* digamma(len(tree_xyz.query_ball_point(data_xyz[i], knn_dis[i], p=np.inf )) -1)
        information_samples[i] += weight[i]* -digamma(len(tree_xz.query_ball_point(data_xz[i], knn_dis[i], p=np.inf )) - 1)
        information_samples[i] += weight[i]* -digamma( np.sum( weight[j] for j in tree_yz.query_ball_point(data_yz[i], knn_dis[i], p=np.inf )) - weight[i])
        information_samples[i] += weight[i]* digamma( np.sum( weight[j] for j in tree_z.query_ball_point(z[i], knn_dis[i], p=np.inf)) - weight[i])
    return np.mean(information_samples)


# CAPACITY ESTIMATORS
# Shannon capacity
def sc(x, y, k=5, bw=0.2, init_weight_option=1, eta=0.5, lamb=100, T=10, method="grad", regularization_type="0" ,th=1e-3):

    def get_obj(adj_x, adj_y, weight):
        N = len(weight)
        ans = 0
        for i in range(N):
            nx = len(adj_x[i]) - 1
            ny = np.sum(weight[j] for j in adj_y[i]) - weight[i]
            ans += -weight[i] * log(nx) / N
            ans += -weight[i] * log(ny) / N
        return ans

    def get_stoch_grad(adj_x, adj_y, weight, i):
        N = len(weight)
        ans = np.zeros(N)
        nx = len(adj_x[i]) - 1
        ny = np.sum(weight[j] for j in adj_y[i]) - weight[i]
        for j in adj_y[i]:
            ans[j] += -weight[i] / (ny * N)
        ans[i] += -(log(nx) + log(ny)) / N #+ weight[i] / (ny * N)
        return ans * np.sqrt(N)

    def get_grad(adj_x, adj_y, weight):
        N = len(weight)
        ans = np.zeros(N)
        for i in range(N):
            nx = len(adj_x[i]) - 1
            ny = np.sum(weight[j] for j in adj_y[i]) - weight[i]
            for j in adj_y[i]:
                ans[j] += -weight[i] / (ny * N)
            ans[i] += -(log(nx) + log(ny)) / N #+ weight[i] / (ny * N)
        return ans


    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"

    # solvers.options['show_progress'] = False

    N = len(x)

    # Sorting x and y based on x
    unsorted_xy = sorted(zip(x, y))
    x = [i for i, _ in unsorted_xy ]
    y = [i for _,i in unsorted_xy ]
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point, k + 1, p=np.inf)[0][k] for point in data]
    adj_x = []
    adj_y = []
    for i in range(N):
        adj_x.append(tree_x.query_ball_point(x[i], knn_dis[i], p=np.inf))
        adj_y.append(tree_y.query_ball_point(y[i], knn_dis[i], p=np.inf))


    if init_weight_option == 0:
        weight = np.ones(N) + nr.normal(0, 0.1, N)
    else:
        k_density = 5
        dx = len(x[0])
        knn_dis = [tree_x.query(point, k_density + 1, p=np.inf)[0][k_density] for point in x]
        density_estimate = np.array([float(k_density) / N / knn_dis[i] ** dx for i in range(len(knn_dis))])
        weight = (1 / density_estimate) / np.mean(1 / density_estimate)
        # kernel = KernelDensity(bandwidth=bw)
        # kernel.fit(x)
        # kde = np.exp(kernel.score_samples(x))
        # weight = (1 / kde).clip(1e-8, np.sqrt(N))
        # weight = weight / np.mean(weight)

    A = np.zeros(N)
    b = 0
    for i in range(N):
        A[i] = (x[i] - np.mean(x)) ** 2
        b += weight[i] * A[i]

    ans = digamma(k) + 2 * log(N - 1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx + dy)

    for i in range(T):

        if method=="grad":
            gradient = get_grad(adj_x, adj_y, weight)
        elif method=="stoch_grad":
            ind = nr.randint(N)
            gradient = get_stoch_grad(adj_x, adj_y, weight, ind)
        else: raise ValueError("Cannot recognize the method")
        # gradient = gradient/np.linalg.norm(gradient)
        # print np.linalg.norm(gradient)

        weight = weight + eta * gradient
        if regularization_type == "1":
            weight = weight - eta * lamb * d_regularizer(weight)
        elif regularization_type == "2":
            weight = d_regularizer_2(weight)
        elif regularization_type == "3":
            weight = d_regularizer_3(weight)

        weight = projection(weight, A, b)

        print(ans + get_obj(adj_x, adj_y, weight))

    print(weight)
    plt.plot(weight)
    plt.show()
    return ans + get_obj(adj_x, adj_y, weight)


def csc(x, y, z, k=5, bw=0.2, init_weight_option=1, eta=0.5, lamb=100, T=10, regularization=True ,th=1e-3):

    def get_obj(adj_xyz, adj_xz, adj_yz, adj_z, weight):
        N = len(weight)
        information_samples = [0 for cnt in range(N)]
        for i in range(N):
            information_samples[i] += weight[i] * digamma( len(adj_xyz[i]) - 1 )
            information_samples[i] += weight[i] * -digamma( len(adj_xz[i]) - 1 )
            information_samples[i] += weight[i] * -digamma( np.sum(weight[j] for j in adj_yz[i]) - weight[i] )
            information_samples[i] += weight[i] * digamma( np.sum(weight[j] for j in adj_z[i]) - weight[i] )
        return np.mean(information_samples)

    def get_grad(adj_xyz, adj_xz, adj_yz, adj_z, weight, i):
        N = len(weight)
        ans = np.zeros(N)
        n_xyz = len( adj_xyz[ind] ) - 1
        n_xz = len(adj_xz[ind]) - 1
        n_yz = np.sum( weight[j] for j in adj_yz[ind] ) - weight[ind]
        n_z = np.sum( weight[j] for j in adj_z[ind] ) - weight[ind]
        for j in adj_yz[ind]:
            ans[j] += -weight[ind] / (n_yz * N)
        for j in adj_z[ind]:
            ans[j] += weight[ind] / (n_z * N)
        ans[i] = ( log(n_xyz) - log(n_xz) - log(n_yz) + log(n_z)) / N #+ weight[ind] / (ny * N)
        return ans * np.sqrt(N)

    assert len(x) == len(y), "Lists should have same length"
    assert len(x) == len(z), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"

    # solvers.options['show_progress'] = False

    N = len(x)
    
    # y = y[np.argsort(x, axis=0)].reshape(N, 1)
    # y = [y[i] for i in np.argsort(x, axis=0)]
    # x = np.sort(x, axis=0)
    data_xyz = np.concatenate((x, y, z), axis=1)
    data_xz = np.concatenate((x, z), axis=1)
    data_yz = np.concatenate((y, z), axis=1)

    tree_xyz = ss.cKDTree(data_xyz)
    tree_xz = ss.cKDTree(data_xz)
    tree_yz = ss.cKDTree(data_yz)
    tree_z = ss.cKDTree(z)

    if init_weight_option == 0:
        weight = np.ones(N) + nr.normal(0, 0.1, N)
    else:
        dx = len(x[0])
        dz = len(z[0])
        k_density = 5
        knn_dis = [tree_xz.query(point, k_density + 1, p=np.inf)[0][k_density] for point in data_xz]
        density_estimate = np.array([float(k_density) / N / knn_dis[i] ** (dx + dz) for i in range(len(knn_dis))])
        weight = (1 / density_estimate) / np.mean(1 / density_estimate)

    A = np.zeros(N)
    b = 0
    for i in range(N):
        A[i] = (x[i] - np.mean(x)) ** 2 + (z[i] - np.mean(z)) ** 2
        b += weight[i] * A[i]

    knn_dis = [tree_xyz.query(point, k + 1, p=np.inf)[0][k] for point in data_xyz]
    adj_xyz = []
    adj_xz = []
    adj_yz = []
    adj_z = []
    for i in range(N):
        adj_xyz.append(tree_xyz.query_ball_point(data_xyz[i], knn_dis[i], p=np.inf))
        adj_xz.append(tree_xz.query_ball_point(data_xz[i], knn_dis[i], p=np.inf))
        adj_yz.append(tree_yz.query_ball_point(data_yz[i], knn_dis[i], p=np.inf))
        adj_z.append(tree_z.query_ball_point(z[i], knn_dis[i], p=np.inf))

    ans = 0 #+ vd(dx) + vd(dy) - vd(dx + dy)

    for i in range(T):
        print(ans + get_obj(adj_xyz, adj_xz, adj_yz, adj_z, weight))
        ind = nr.randint(N)
        gradient = get_grad(adj_xyz, adj_xz, adj_yz, adj_z, weight, ind)
        # gradient = gradient/np.linalg.norm(gradient)
        weight = weight + eta * gradient
        if regularization:  weight = weight - eta * lamb * d_regularizer(weight)
        weight = projection(weight, A, b)

    return ans + get_obj(adj_xyz, adj_xz, adj_yz, adj_z, weight)


def d_regularizer(weight):
    N = len(weight)
    ans = np.zeros(N)
    for i in range(len(weight) - 1):
        ans[i] += weight[i] - weight[i + 1]
        ans[i + 1] += weight[i + 1] - weight[i]
    return ans / N


def d_regularizer_2(weight,window_size=2):
    N = len(weight)
    ans = np.zeros(N)
    for i in range(N):
        ans[i] = np.sum( weight[ max(0,i-window_size/2): min(N,i+window_size/2) ] ) / float(window_size)
    return ans


def d_regularizer_3(weight,number_of_sections=100):
    N = len(weight)
    ans = np.zeros(N)
    section_len = N / number_of_sections
    for i in range(number_of_sections):
        value = np.mean( weight[i*section_len:(i+1)*section_len] )
        for j in range(i*section_len,(i+1)*section_len):
            ans[j] = value
    return ans


def projection(w, A, b):
    N = len(w)
    w = w.reshape((len(w), 1))
    A = A.reshape((len(w), 1))
    x = np.dot(np.linalg.inv(np.dot(A.transpose(), A)), np.dot(A.transpose(), w) - b)
    if x > 0:
        w = w - np.dot(A, x)
    w = w.clip(1e-6).reshape(N)
    return w / np.mean(w)



# N = 10000
# mi_1 = []
# umi_1 = []
# umi_alternate_1 = []
# umi_10 = []
# umi_alternate_10 = []
#
# max_range = np.arange(.01,1.01,.02)
#
# for u_max in max_range:
#
#     X = [ [ uniform(0,1)**1 ] for cnt in range(N) ]
#     Y = [ [ ( X[cnt][0] + uniform(0,u_max) ) ] for cnt in range(N) ]
#     mi_1.append( mi(X,Y) )
#     umi_1.append( umi(X,Y, density_estimation_method="knn") )
#     umi_alternate_1.append( alternate_umi(X,Y, density_estimation_method="knn") )
#
#     X = [[uniform(0, 1)**10 ] for cnt in range(N)]
#     Y = [[(X[cnt][0] + uniform(0, u_max))] for cnt in range(N)]
#     umi_10.append( umi(X, Y, density_estimation_method="knn"))
#     umi_alternate_10.append( alternate_umi(X, Y, density_estimation_method="knn") )
#
# plt.plot(max_range, mi_1, label="MI for uniform" )
# plt.plot(max_range, umi_1, "r*", label="UMI for uniform" )
# plt.plot(max_range, umi_alternate_1, "c*", label="Alternate UMI for uniform" )
# plt.plot(max_range, umi_10, "r+", label="UMI for uniform^10" )
# plt.plot(max_range, umi_alternate_10, "c+", label="alternate UMI for uniform^10" )
# plt.legend()
# plt.show()


# N = 1000
# u_max = .2
# X = [ [ uniform(0,1)**1] for cnt in range(N) ]
# Y = [ [ ( X[cnt][0] + uniform(0,u_max) ) % 1 ] for cnt in range(N) ]
# print mi(X,Y)
# print umi(X,Y,density_estimation_method="knn")
# print sc(X,Y, init_weight_option=0, regularization_type="3", method="grad", eta=.2, lamb=100, T=2000)

# Z = [ [ uniform(0,1)**5 ] for cnt in range(N) ]
# Y = [ [ Y[cnt][0] + Z[cnt][0] ] for cnt in range(N) ]

# print cmi(X,Y,Z)
# print cumi(X,Y,Z, density_estimation_method="knn")
# print csc(X,Y,Z, init_weight_option=1, regularization=False, eta=.02, T=N)

# Implemented ONLY for 1-d vectors, for now
def cumi_partitioning(x, y, z, no_partitions):

    assert len(x) == len(y), "Lists should have same length"
    assert len(x) == len(z), "Lists should have same length"

    N = len(x)

    x_min = min([i[0] for i in x])
    x_max = (1+10e-10)*max([i[0] for i in x])
    y_min = min([i[0] for i in y])
    y_max = (1+10e-10)*max([i[0] for i in y])
    z_min = min([i[0] for i in z])
    z_max = (1+10e-10)*max([i[0] for i in z])

    x_bin_len = float(x_max-x_min) / no_partitions
    y_bin_len = float(y_max-y_min) / no_partitions
    z_bin_len = float(z_max-z_min) / no_partitions

    p_xyz = {}
    p_xz = {}
    for ix in range(no_partitions):
        for iz in range(no_partitions):
            p_xz[(ix, iz)] = 0
            for iy in range(no_partitions):
                p_xyz[(ix, iy, iz)] = 0

    for i in range(N):
        ix = np.nan
        iy = np.nan
        iz = np.nan
        for id in range(no_partitions):
            if (x_min + x_bin_len*id) <= x[i][0] and x[i][0] < (x_min + x_bin_len*(1+id)): ix = id
            if (y_min + y_bin_len*id) <= y[i][0] and y[i][0] < (y_min + y_bin_len*(1+id)): iy = id
            if (z_min + z_bin_len*id) <= z[i][0] and z[i][0] < (z_min + z_bin_len*(1+id)): iz = id
        p_xyz[(ix, iy, iz)] += 1
        p_xz[(ix, iz)] += 1

    p_y_given_xz = {}
    for ix in range(no_partitions):
        for iz in range(no_partitions):
            p_xz[(ix, iz)] /= float(N)
            for iy in range(no_partitions):
                p_xyz[(ix, iy, iz)] /= float(N)
                if p_xyz[(ix, iy, iz)] == 0:
                    p_y_given_xz[(ix, iy, iz)]=0.0
                    continue
                p_y_given_xz[(ix, iy, iz)] = p_xyz[(ix, iy, iz)] / float(p_xz[(ix, iz)])

    pu_z = {}
    pu_yz = {}
    for iz in range(no_partitions):
        pu_z[iz] = 0.0
        for iy in range(no_partitions):
            pu_yz[(iy, iz)] = 0.0
            for ix in range(no_partitions):
                pu_yz[(iy, iz)] += p_y_given_xz[(ix, iy, iz)]
                pu_z[iz] += p_y_given_xz[(ix, iy, iz)]

    pu_y_given_z = {}
    for iz in range(no_partitions):
        for iy in range(no_partitions):
            if pu_yz[(iy, iz)] == 0:
                pu_y_given_z[(iy, iz)] = 0.0
                continue
            pu_y_given_z[(iy, iz)] = pu_yz[(iy, iz)] / pu_z[iz]

    ans = 0.0
    for ix in range(no_partitions):
        for iz in range(no_partitions):
            for iy in range(no_partitions):
                if p_xyz[ (ix, iy, iz) ]== 0: continue
                ans += 1.0/no_partitions**2 * p_y_given_xz[ (ix, iy, iz) ] * np.log( p_y_given_xz[ (ix, iy, iz) ]/pu_y_given_z[ (iy, iz) ] )

    return ans



# N = 5000
# alpha = .99; alpha_tilda=np.sqrt(1-alpha**2)
# X = [[gauss(0,1)] for i in range(N)]
# Y = [[alpha*X[i][0]+alpha_tilda*gauss(0,1)] for i in range(N)]
# Z = [[alpha*Y[i][0]+alpha_tilda*gauss(0,1)] for i in range(N)]
# print mi(X,Y)
# print mi_alternate(X,Y)

# N = 1000
# u_max = .2
# X = [ [ uniform(0,1)**1 ] for cnt in range(N) ]
# Y = [ [ ( X[cnt][0] + uniform(0,u_max) ) ] for cnt in range(N) ]
# print mi(X,Y)
# print umi(X,Y, density_estimation_method="knn")
# # print alternate_umi(X,Y, density_estimation_method="knn")
#
# X = [[uniform(0, 1)**15 ] for cnt in range(N)]
# Y = [[(X[cnt][0] + uniform(0, u_max))] for cnt in range(N)]
# print umi(X, Y, density_estimation_method="knn")
# # print alternate_umi(X, Y, density_estimation_method="knn")

# N = 1000
# u_max = .2
#
# n=5
# X = [ [uniform(0,1)**n] for i in range(N) ]
# Z = [ [uniform(0,1)**n] for i in range(N) ]
# Y = [ [ (X[i][0] +Z[i][0] +uniform(0,u_max)) ] for i in range(N) ]
#
# # print cumi(X,Y,Z)
# print cumi_partitioning(X,Y,Z,no_partitions=25)
