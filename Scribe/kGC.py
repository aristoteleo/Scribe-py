import numpy as np
import pandas
from scipy.stats import pearsonr
from multiprocessing import Pool
from .granger import granger

def __individual_granger(id1, id2, arr, maxlag):
    return (id1, id2, granger(arr, maxlag=maxlag, addconst=True, verbose=False)[maxlag][0]["lrtest"][0])


def __individual_kernel_granger(id1, id2, arr, type, param, maxlag):
    return (id1, id2, KernelGrangerCausality(arr, type, param, maxlag))

#######################################################
#######################################################
def generate_snippets_present(x_timeseries,m):
    '''
    This function takes the time vector x_vec and generates the matrices x and X
    INPUT:
            x_timeseries : the T-by-nvar array containing nvar time-series of length T
            m: The order of the lags used here (The length oh each history vector in X)
    OUTPUT:
            x: present values of the vector x_vec, shape N-by-nvar
            snippets: m-snippets (past history up to m samples) corresponding to each of x values, shape m-by-nvar-by-N
    '''

    T, nvar = x_timeseries.shape

    assert m<T, "The history length has to be smaller than the time-series total length"
    N = T - m

    x = x_timeseries[m:,:]
    snippets = np.nan * np.ones((m,nvar,N))
    for cnt in range(m,T): snippets[:,:,cnt-m] = x_timeseries[cnt-m:cnt,:]

    ## Normalize the output vectors
    # x =  (x-np.mean(x,0)) / np.std(x,0)
    # snippets = snippets - np.mean(snippets,0)

    return x, snippets


#######################################################
#######################################################
def kernel(X,type,par):
    '''
    This function takes the m-by-N array X as an input argument and outputs the pairwise kernel matrix K for the given kernel type
    INPUT :
            X: The m*N array containing N samples of length m
            type: The type of the kernel used:
                            "g" for Gaussian
                            "l" for Linear
                            "p" for Polynomial
                            "h" for Homogeneous polynomial
            par: the parameter of the kernel
                            for Gaussian: the bandwidth of the kernel
                            for Linear: (void)
                            for Polynomial/HomogeneousPolynomial: the degree of the polynomial
    OUTPUT :
            K: The N-by-N Kernel array for which K(i,j)=k(X[i],X[j])
    '''

    A = np.matmul(X.transpose(),X) # The matrix A contains pairwise inner products: A[i,j] = < X[i], X[j] >
    if type[0].lower()=="g": B = np.ones((A.shape[0],1)) * np.diag(A)

    # Apply the kernel to the data
    if type[0].lower()=="l": K = A
    elif type[0].lower()=="p": K = (1 + A)**par
    elif type[0].lower()=="h": K = A**par
    elif type[0].lower()=="g": K = np.exp( (2.0*A-B-np.transpose(B))/(2*par**2) )
    else: raise ValueError("Invalid kernel type.")


    K_aux = np.ones((K.shape[0],1)) * np.mean(K,0)
    K = K - K_aux - K_aux.transpose() - np.mean(K)

    # TODO: For guassian, do the K* fix

    return K


#######################################################
#######################################################
def calculate_p(K):

    eigen_vals, eigen_vecs = np.linalg.eig(K)
    assert max(abs(np.imag(eigen_vals)))<1e-8, "All the eigen values have to be real."
    P = np.zeros((K.shape))
    for i in range(len(eigen_vals)):
        if abs(np.real(eigen_vals[i]))>1e-8:
            a = np.real(eigen_vecs[:, i])
            P += np.matmul(a.reshape(-1,1),a.reshape(1,-1))

    return P


#######################################################
#######################################################
def extract_ti_vectors(K_prime, P):

    K_tilde = K_prime - np.matmul(P,K_prime) - np.matmul(K_prime,P) + np.matmul(np.matmul(P,K_prime),P)

    eigen_vals, eigen_vecs = np.linalg.eig(K_tilde)
    assert max(abs(np.imag(eigen_vals))) < 1e-8, "All the eigen values have to be real."

    ti_vectors = []
    for i in range(len(eigen_vals)):
        if abs(np.real(eigen_vals[i])) > 1e-8:
            a = np.real(eigen_vecs[:, i])
            ti_vectors.append(a)

    return np.array(ti_vectors).transpose()

#######################################################
#######################################################
def calculate_delta(Ti, y_vec, b_thres):
    '''
    This function takes the matrix of column ti vectors and y_vec, and calculates sum of square of pairwise correlations.
    INPUT:
            Ti: The matrix of ti vectors
            y_vec: defined as x-P*x
            b_thres: Threshold used in Bonferroni correction
    OUTPUT:
            delta: Causality index from y to x
    '''
    m = Ti.shape[1]
    r_values = [ pearsonr(Ti[:,i],y_vec) for i in range(m) ]

    delta = np.sum( [ r_values[i][0]**2 for i in range(m) if r_values[i][1]<b_thres/m ] )

    return delta


#######################################################
#######################################################
def pairwise_kgc(x,X,Z, kernel_type, kernel_param):
    '''
    This function takes matrices x, X and Z and calculates the pairwise Kernel Granger Causality score from the "cause"
    variable to the "target" variable.
    The target variable is determined by vector x, and the cause variable is the variable that exists in Z but does
    not exist in X
    INPUT:
            x: The N-by-1 vector of the present values of the target variable "X"
            X: The (c+1)*m-by-N matrix including the m snippets of the variable "X" and c conditional variables
               (usually c=nvar-2, excepting the cause variable "Y")
            Z: The nvar*m-by-N matrix including the m snippets of all variables (including the cause y)
    OUTPUT:
            delta: causality score
    '''

    # Generate the necessary Kernel matrices
    K = kernel(X, type=kernel_type, par=kernel_param)
    K_prime = kernel(Z, type=kernel_type, par=kernel_param)
    P = calculate_p(K)
    Ti = extract_ti_vectors(K_prime, P)

    # Calculating the causality score
    delta = calculate_delta(Ti, x-np.matmul(P,x), b_thres=.05)
    return delta


########################################################################################################################
########################################################################################################################
########################################################################################################################
def KernelGrangerCausality(x_timeseries, kernel_type, kernel_param, maxlag=1):
    '''
    KERNEL GRANGER CAUSALITY
    This module takes the a T-by-nvar array consisting of nvar time-series of length T and calculates the causality from
    each cause variable "i" to each target variable "j" conditioning on the past of all the other variables inherently.
    INPUT:
            x_timeseries: The T-by-nvar array of input time-series
            kernel_type: The type of the kernel used:
                            "g" for Gaussian
                            "l" for Linear
                            "p" for Polynomial
                            "h" for Homogeneous polynomial
            kernel_param: the parameter of the kernel
                            for Gaussian: the bandwidth of the kernel
                            for Linear: (void)
                            for Polynomial/HomogeneousPolynomial: the degree of the polynomial
    OUTPUT:
            causality_scores: The nvar-by-nvar array containing the directional causality scores for all the variables
                              causality_scores[i,j] = causality score from cause "i" to target "j"
    '''


    nvar = x_timeseries.shape[1] # Number of variables
    N = x_timeseries.shape[0] - maxlag # Number of samples (samples: present values + corresponding past snippets of length m)
    causality_scores = np.nan * np.ones((nvar,nvar))

    # First generate past snippets, x and Z matrix
    x, snippets = generate_snippets_present(x_timeseries, maxlag)
    Z = snippets.transpose(1,0,2).reshape(nvar * maxlag, N) # All the snippets for all the variables

    # # Then let's do the pairwise causality test
    # for i in range(nvar):
    #     # X: All the snippets for all the variables except the cause "i"
    #     X = snippets[:,np.arange(nvar)!=i,:].transpose(1,0,2).reshape((nvar-1) * maxlag, N)
    #     for j in range(nvar):
    #         if i==j: continue
    #         causality_scores[i,j]=pairwise_kgc(x[:,j], X, Z, kernel_type, kernel_param)

    i=0
    j=1
    # X: All the snippets for all the variables except the cause "i"
    X = snippets[:, np.arange(nvar) != i, :].transpose(1, 0, 2).reshape((nvar - 1) * maxlag, N)
    causality_score = pairwise_kgc(x[:, j], X, Z, kernel_type, kernel_param)

    return causality_score


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
# from random import gauss
# x_timeseries = [ [gauss(0,1) for _ in range(1000)] ]
# x_timeseries.append( [0]+[x_timeseries[0][i]**3+gauss(0,.1) for i in range(1000)][:-1]  )
# x_timeseries.append( [0]+[x_timeseries[1][i]**1+gauss(0,.1) for i in range(1000)][:-1]  )
# x_timeseries = np.array(x_timeseries).transpose()
# # pdb.set_trace()
#
# print KernelGrangerCausality(x_timeseries, "l", 4, maxlag=1)

##############################################################################
# Run pairwise Kernel Granger over the data for the given delay
def kernel_granger(self, type, par, maxlag=1, number_of_processes=1):

    self.kernel_granger_results = pandas.DataFrame({node_id: [np.nan for i in self.node_ids] for node_id in self.node_ids}, index=self.node_ids)
    if number_of_processes > 1: temp_input = []

    for id1 in self.node_ids:
        for id2 in self.node_ids:

            if id1 == id2: continue

            arr = np.array([self.expression_concatenated.loc[id2],self.expression_concatenated.loc[id1]]).transpose()
            if number_of_processes == 1:
                self.kernel_granger_results.loc[id1, id2] = __individual_kernel_granger((id1, id2, arr, type, par, maxlag))[2]
            else:
                temp_input.append((id1, id2, arr, maxlag))

    if number_of_processes > 1:
        tmp_results = Pool(number_of_processes).map(__individual_granger, temp_input)
        for t in tmp_results: self.kernel_granger_results.loc[t[0], t[1]] = t[2]

    return self.kernel_granger_results
