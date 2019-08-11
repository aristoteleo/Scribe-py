##############################################################################
# Run pairwise Granger over the data for the given delay
import statsmodels
def granger(self, maxlag=1, number_of_processes=1):

    self.granger_results = pandas.DataFrame({node_id: [np.nan for i in self.node_ids] for node_id in self.node_ids}, index=self.node_ids)
    if number_of_processes > 1: temp_input = []

    for id1 in self.node_ids:
        for id2 in self.node_ids:

            if id1 == id2: continue

            arr = np.array([self.expression_concatenated.loc[id2],self.expression_concatenated.loc[id1]]).transpose()
            if number_of_processes == 1:
                self.granger_results.loc[id1, id2] = __individual_granger((id1, id2, arr, maxlag))[2]
            else:
                temp_input.append((id1, id2, arr, maxlag))

    if number_of_processes > 1:
        tmp_results = Pool(number_of_processes).map(__individual_granger, temp_input)
        for t in tmp_results: self.granger_results.loc[t[0], t[1]] = t[2]

    return self.granger_results