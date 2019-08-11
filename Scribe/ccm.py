##############################################################################
# Run pairwise CCM over the data for the given delay
from .pyccm import *

def ccm(self, tau=1, E=None, periods=1, number_of_processes=1):
    """Calculate causal network by CCM (Convergent Cross Mapping)
    Uses a leave one out cross validation to calculate the predictability of X|Y and Y|X. It can be used to determine if
    one time series contain the necessary dynamic information to recover the influence of another causal variable.

    Arguments
    ---------
        self: 'class causal_model object'
            An instance of a causal_model class object. This object can be converted from an AnnData object through
            load_anndata function.
        tau: 'np.ndarray'
            Lag to use for time delay embedding.
        E: 'np.ndarray'
            Number of lags to include
        periods: `int` (default: 1)
            Number of periods.
        number_of_processes: `int` (default: 1)
            Number of cores to use.
                
    Returns
    -------
    ccm_results: 'pd.core.frame.DataFrame'
        The casual network inferred from CCM algorithm.
    """

    self.ccm_results = pandas.DataFrame({node_id: [np.nan for i in self.node_ids] for node_id in self.node_ids}, index=self.node_ids)
    if number_of_processes > 1: temp_input = []

    for id1 in self.node_ids:
        for id2 in self.node_ids:

            if id1 == id2: continue

            arr = np.array([(self.expression_concatenated.loc[id1].values[i], self.expression_concatenated.loc[id2].values[i])
                            for i in range(len(self.expression_concatenated.loc[id1]))], dtype=[(id1, np.float), (id2, np.float)])


            if number_of_processes == 1 :
                self.ccm_results.loc[id1, id2] = __individual_ccm((id1, id2, arr, tau, E, periods))[2]
            else:
                temp_input.append((id1, id2, arr, tau, E, periods))

    if number_of_processes > 1:
        tmp_results = Pool(number_of_processes).map(__individual_ccm, temp_input)
        for t in tmp_results: self.ccm_results.loc[t[0], t[1]] = t[2]\

    return self.ccm_results
