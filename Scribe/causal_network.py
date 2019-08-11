import warnings
import pandas
import numpy as np
from multiprocessing import Pool

from copy import deepcopy
from .information_estimators import mi, cmi, cumi        #######added a '.'
from .other_estimators import corr                         #####################3

########################################################################################################################
########################################################################################################################
def _rdi_causal_model__individual_mi(id1, id2, x, y):
    return (id1, id2, mi(x, y))

def _rdi_causal_model__individual_corr(id1, id2, x, y):
    return (id1, id2, corr(x, y)[0])

def _rdi_causal_model__individual_cmi(id1, id2, x, y, z, uniformization, differential_mode):
    if uniformization==True: return (id1, id2, cumi(x, y, z, normalization=differential_mode))
    else: return (id1, id2, cmi(x, y, z, normalization=differential_mode))

def _rdi_causal_model__individual_ccm(id1, id2, arr, tau, E, periods):
    return (id1, id2, ccm(arr, tau=tau, E=E, periods=periods)[id1 + '->' + id2])

def _rdi_causal_model__individual_granger(id1, id2, arr, maxlag):
    return (id1, id2, granger(arr, maxlag=maxlag, addconst=True, verbose=False)[maxlag][0]["lrtest"][0])

def _rdi_causal_model__individual_kernel_granger(id1, id2, arr, type, param, maxlag):
    return (id1, id2, KernelGrangerCausality(arr, type, param, maxlag))

class causal_model(object):
    ''' This class defines an object for causal inference.
    Upon creation, each object reads and loads the gene expression data, and later on it can calculate the pairwise
    causal score for each pair of genes, and returns the ROC results if given the ground-truth. '''

    #############################################################################
    def __init__(self, expression=None):

        if expression==None: warnings.warn(" WARNING: No expression data argument given. if you intend to load the data from a file, call the method 'read_expression_file'.")
        else:
            self.expression = deepcopy(expression)
            self.expression_raw = expression
        self.rdi_results = None
        self.crdi_results = None

    #############################################################################
    def read_expression_file(self, expression_path, data_mode, verbose=False, genotype_path=None, phenotype_path=None):
        '''
        A module to read the data from the files and store them in a pandas data-frame

        The expression data file should be organized as this:
        For single run mode:
          1st line: "GENE_ID" + cell IDs
          other lines: first column: Genes ID. Other columns: genes expression in cells (pseudo-times)
        For multi run mode:
          1st line: "GENE_ID" + "RUN_ID" + cell IDs
          other lines: first column: Genes ID. Run ID. Other columns: genes expression in cells (pseudo-times)
        '''

        if verbose==True: print("\nLoading data... ",)


        if data_mode == "single_run":
            self.expression = pandas.read_table(expression_path, delimiter="\t", header=0, index_col="GENE_ID")
            self.expression_raw = deepcopy(self.expression)
            self.node_ids = self.expression.index
            self.expression_concatenated = self.expression
        elif data_mode == "multi_run":
            self.expression = pandas.read_table(expression_path, delimiter="\t", header=0, index_col=["GENE_ID", "RUN_ID"])
            self.expression_raw = deepcopy(self.expression)
            self.node_ids = self.expression.index.levels[0]
            self.run_ids = self.expression.index.levels[1]
            self.expression_concatenated = pandas.concat( [ pandas.DataFrame(self.expression.loc[node_id].values.reshape(1, -1), index=[node_id]) for node_id in self.node_ids ] )
        else:
            raise ValueError("Data mode invalid")

        if verbose==True:
            print("DONE.")
            if data_mode == "single_run":
                print("A total number of", self.expression.shape[0], "genes, each containing", self.expression.shape[1], "pseudo-times were read.\n")
            else:
                pass
                print("A total number of", len(self.expression.index.levels[0]), "genes, each gene containing", len(
                    self.expression.index.levels[1]), "runs and each run containing at most", self.expression.shape[1], "pseudo-times were read.\n")

    #############################################################################
    def restore_the_raw_expression_data(self):
        '''A method to return the original gene expressions, in case any modifications made'''

        self.expression = deepcopy(self.expression_raw)

    ##############################################################################
    def subset_genes(self,subset):
        ''' This subroutine filters the input genes based on a given subset '''

        if len(self.expression.index.names) == 1:
            self.expression = self.expression.loc[subset, :]
        else:
            index_set_tmp = [ index for index in self.expression.index if index[0] in subset]
            self.expression = self.expression.loc[index_set_tmp, :]
        self.node_ids = subset

    ##############################################################################
    def subset_runs(self,subset):
        ''' This subroutine filters the input runs based on a given subset of run ids '''

        if len(self.expression.index.names) == 1:
            warnings.warn("WARNING: Cannot perform run-subsetting on a single-run mode data.")
        else:
            index_set_tmp = [ index for index in self.expression.index if index[1] in subset]
            self.expression = self.expression.loc[index_set_tmp, :]
            self.expression_concatenated = pandas.concat( [pandas.DataFrame(self.expression.loc[node_id].values.reshape(1, -1), index=[node_id]) for node_id in self.node_ids])
            self.run_ids = subset

    ##############################################################################
    def smooth(self, window):
        ''' This subroutine smoothens the gene expressions based on a given window size '''

        self.expression = self.expression_raw.rolling(window=window,axis=1).mean()
        self.expression = self.expression.dropna(axis=1,how="all")

    ##############################################################################
    def normalize(self):
        '''This subroutine normalizes the gene expressions'''

        self.expression = self.expression.sub(self.expression.mean(axis=1),axis=0)
        self.expression = self.expression.div(self.expression.std(axis=1), axis=0)
        self.expression = self.expression.fillna(0)

    ##############################################################################
    def rdi(self, delays, number_of_processes=1, uniformization=False, differential_mode=False):
        ''' Run pairwise Restricted Directed Information over the data for the given delay list
            Input parameters:
                delays: the list of delays for which the RDI values will be assessed
                number_of_processes: The number of processes for the purpose of parallel-processing
                uniformization: If True, uCMI will be used instead of CMI
                differential mode: If True, X(t) will be replaced by X(t)-X(t-1)
            Output:
                self.rdi_results:
                A dictionary of length len(delays)+1, including m-by-m pandas data frames consisting of pairwise RDI scores
                The results include the pairwise results for each delay, as well as the maximal pairwise scores across all the delays
                stored in self.rdi_results["MAX"].
        '''
        self.rdi_results = {}

        for delay in delays:
            #print(self.node_ids)
            #print(delay)
            self.rdi_results[delay] = pandas.DataFrame({node_id: [np.nan for i in self.node_ids] for node_id in self.node_ids}, index=self.node_ids)###made a nan fit matrix for rdi delay
            if number_of_processes>1: temp_input = []

            for id1 in self.node_ids:
                for id2 in self.node_ids:

                    if id1 == id2: continue

                    if len(self.expression.index.names) == 1:  # If the data consists of only a single run of the process

                        x = [[i] for i in  self.expression.loc[id1].dropna()][:-delay]

                        y_minus_1 = [[j] for j in self.expression.loc[id2].dropna()][delay - 1:-1]
                        if differential_mode == False:
                            y = [[j] for j in self.expression.loc[id2].dropna()][delay:]
                        elif differential_mode == True:
                            y = [[self.expression.loc[id2].dropna()[j - 1] - self.expression.loc[id2].dropna()[j]] for j in range(1, len(self.expression.loc[id2].dropna()))][delay - 1:]

                        if number_of_processes == 1:
                            self.rdi_results[delay].loc[id1, id2] = (_rdi_causal_model__individual_cmi(id1, id2, x, y, y_minus_1, uniformization, differential_mode))[2]
                        else:
                            temp_input.append(id1, id2, x, y, y_minus_1, uniformization, differential_mode)

                    else:  # If the data consists of multiple runs of the process
                        x = []
                        y = []
                        y_minus_1 = []

                        for run_id in self.run_ids:
                            #print("run_id: "+str(run_id))
                            x += [[i] for i in self.expression.loc[id1, run_id].dropna()][:-delay]
                            #print(x)
                            y_minus_1 += [[k] for k in self.expression.loc[id2, run_id].dropna()][delay - 1:-1]
                            if differential_mode == False:
                                y += [[j] for j in self.expression.loc[id2,run_id].dropna()][delay:]
                            elif differential_mode == True:
                                y += [[self.expression.loc[id2,run_id].dropna()[j-1]-self.expression.loc[id2,run_id].dropna()[j]] for j in range(1,len(self.expression.loc[id2,run_id].dropna()))][delay-1:]

                        if number_of_processes == 1:
                            self.rdi_results[delay].loc[id1, id2] = (_rdi_causal_model__individual_cmi(id1, id2, x, y, y_minus_1, uniformization, differential_mode))[2]
                        else:
                            temp_input.append((id1, id2, x, y, y_minus_1 ,uniformization, differential_mode))

            if number_of_processes>1 :
                tmp_results = Pool(number_of_processes).map((_rdi_causal_model__individual_cmi), temp_input)
                for t in tmp_results: self.rdi_results[delay].loc[t[0], t[1]] = t[2]

        self.rdi_results["MAX"] = self.__extract_max_rdi_value_delay()[0]                        #########################not exist      self.__extract_max_rdi_value_delay()
        #print(self.rdi_results)

        return self.rdi_results

    ##############################################################################
    def crdi(self, L=1, number_of_processes=1, uniformization=False, differential_mode=False):
        ''' Run pairwise Conditional Restricted Directed Information (cRDI) over the data for the given number of conditional variables
            Input parameters:
                L: The number of conditional variables for which the conditional RDI values will be assessed
                   The L variables will be chosen based on the top RDI (unconditional) incoming values to each variable
                number_of_processes: The number of processes for the purpose of parallel-processing
                uniformization: If True, uCMI will be used instead of CMI
                differential mode: If True, X(t) will be replaced by X(t)-X(t-1)
            Output:
                self.rdi_results:
                A dictionary of length len(delays)+1, including m-by-m pandas data frames consisting of pairwise RDI scores
                The results include the pairwise results for each delay, as well as the maximal pairwise scores across all the delays
                stored in self.crdi_results["MAX"].
        '''

        if self.rdi_results==None:
            warnings.warn("WARNING: The method first needs RDI to be run, RUNNING RDI NOW... delay set to [1]")
            self.rdi(delays=[1], number_of_processes=number_of_processes, uniformization=uniformization, differential_mode=differential_mode)

        self.crdi_results = pandas.DataFrame({node_id: [np.nan for i in self.node_ids] for node_id in self.node_ids}, index=self.node_ids)
        if number_of_processes > 1: temp_input = []

        [max_rdi_value, max_rdi_delay] = self.__extract_max_rdi_value_delay()
        [top_incoming_nodes, top_incoming_delays, top_incoming_values] = self.__extract_top_incoming_nodes_delays(
            max_rdi_value, max_rdi_delay, k_nodes=L)

        for id1 in self.node_ids:
            for id2 in self.node_ids:
                if id1 == id2: continue

                top_incoming_nodes_tmp = deepcopy(top_incoming_nodes[id2])
                top_incoming_delays_tmp = deepcopy(top_incoming_delays[id2])

                if id1 in top_incoming_nodes[id2]:
                    del top_incoming_nodes_tmp[top_incoming_nodes[id2].index(id1)]
                    del top_incoming_delays_tmp[top_incoming_nodes[id2].index(id1)]
                else:
                    del top_incoming_nodes_tmp[top_incoming_values[id2].index(min(top_incoming_values[id2]))]
                    del top_incoming_delays_tmp[top_incoming_values[id2].index(min(top_incoming_values[id2]))]

                if len(self.expression.index.names) == 1:  # If the data consists of only a single run of the process
                    delay = max_rdi_delay.loc[id1, id2]
                    tau = max(top_incoming_delays_tmp + [delay])
                    total_length = len(self.expression.loc[id2].dropna()) - tau
                    x = [[i] for i in self.expression.loc[id1].dropna()][tau - delay:tau - delay + total_length]
                    yz = self.expression.loc[id2].dropna()[tau - 1:tau - 1 + total_length].values.reshape(-1,1)
                    for i in range(L):
                        id3 = top_incoming_nodes_tmp[i]
                        delay_id3 = top_incoming_delays_tmp[i]
                        yz = np.concatenate( ( self.expression.loc[id3].dropna()[tau - delay_id3:tau - delay_id3 + total_length].values.reshape(-1,1), yz), axis=0)
                    if differential_mode == False:
                        y = [[i] for i in self.expression.loc[id2].dropna()][tau:tau + total_length]
                    elif differential_mode == True:
                        y = [[self.expression.loc[id2].dropna()[j-1]-self.expression.loc[id2].dropna()[j]] for j in range(1,len(self.expression.loc[id2].dropna()))][tau-1:tau + total_length]

                    if number_of_processes == 1:
                        self.crdi_results.loc[id1, id2] = (_rdi_causal_model__individual_cmi(id1, id2, x, y, yz, uniformization, differential_mode))[2]
                    else:
                        temp_input.append((id1, id2, x, y, yz, uniformization, differential_mode))

                else:  # If the data consists of multiple runs of the process

                    x = []
                    y = []
                    yz = []
                    for run_id in self.run_ids:
                        delay = max_rdi_delay.loc[id1, id2]
                        tau = max(top_incoming_delays_tmp + [delay])
                        total_length = len(self.expression.loc[id2,run_id].dropna()) - tau
                        x += [[i] for i in self.expression.loc[id1,run_id].dropna()][tau - delay:tau - delay + total_length]
                        yz_tmp = [[j] for j in self.expression.loc[id2,run_id].dropna()][tau - 1:tau - 1 + total_length]
                        if differential_mode==False:
                            y += [[j] for j in self.expression.loc[id2, run_id].dropna()][tau:tau + total_length]
                        elif differential_mode==True:
                            y += [[self.expression.loc[id2,run_id].dropna()[j-1]-self.expression.loc[id2,run_id].dropna()[j]] for j in range(1,len(self.expression.loc[id2,run_id].dropna()))][tau-1:tau + total_length]

                        for i in range(L):
                            id3 = top_incoming_nodes_tmp[i]
                            delay_id3 = top_incoming_delays_tmp[i]
                            #print(([[j] for j in self.expression.loc[id3,run_id].dropna()][tau-delay_id3:tau-delay_id3+total_length], yz_tmp))
                            yz_tmp = list(np.concatenate(([[j] for j in self.expression.loc[id3,run_id].dropna()][tau-delay_id3:tau-delay_id3+total_length], yz_tmp), axis=1))
                        yz += yz_tmp    #######################an array list

                    if number_of_processes == 1:
                        self.crdi_results.loc[id1, id2] = (_rdi_causal_model__individual_cmi(id1, id2, x, y, yz, uniformization, differential_mode))[2]
                    else:
                        temp_input.append((id1, id2, x, y, yz, uniformization, differential_mode))

        if number_of_processes > 1:
            tmp_results = Pool(number_of_processes).map((_rdi_causal_model__individual_cmi), temp_input)
            for t in tmp_results: self.crdi_results.loc[t[0], t[1]] = t[2]

        return self.crdi_results

    ########################################
    # An auxiliary private module which will extract the delays corresponding to the max rdi value calculated.
    # This module is used by CRDI
    def __extract_max_rdi_value_delay(self):
        '''An auxiliary private module which will extract the delays corresponding to the max rdi value calculated.This module is used by CRDI'''
        max_rdi_value = pandas.DataFrame({node_id: [-np.inf for i in self.node_ids] for node_id in self.node_ids}, index=self.node_ids)
        max_rdi_delay = pandas.DataFrame({node_id: [np.nan for i in self.node_ids] for node_id in self.node_ids}, index=self.node_ids,
                                         dtype=np.int)

        for delay in self.rdi_results.keys():
            if delay == "MAX": continue
            for id1 in self.node_ids:
                for id2 in self.node_ids:

                    if id1 == id2: continue
                    if self.rdi_results[delay].loc[id1, id2] > max_rdi_value.loc[id1, id2]:
                        max_rdi_value.loc[id1, id2] = self.rdi_results[delay].loc[id1, id2]
                        max_rdi_delay.loc[id1, id2] = delay

        return max_rdi_value, max_rdi_delay

    ########################################
    # An auxiliary private module which for each node, will extract the incoming nodes with the highest rdi values and
    # their corresponding delays. This module is used by CRDI
    def __extract_top_incoming_nodes_delays(self, max_rdi_values, max_rdi_delays, k_nodes):
        '''An auxiliary private module which for each node, will extract the incoming nodes with the highest rdi values and their corresponding delays. This module is used by CRDI'''
        top_incoming_nodes = {destination_id: [None for i in range(k_nodes + 1)] for destination_id in self.node_ids}
        top_incoming_delays = {destination_id: [np.nan for i in range(k_nodes + 1)] for destination_id in self.node_ids}
        top_incoming_values = {destination_id: [-np.inf for i in range(k_nodes + 1)] for destination_id in self.node_ids}

        for destination_id in self.node_ids:
            for source_id in self.node_ids:

                if destination_id == source_id: continue

                if max_rdi_values.loc[source_id, destination_id] > min(top_incoming_values[destination_id]):
                    min_index = top_incoming_values[destination_id].index(min(top_incoming_values[destination_id]))
                    top_incoming_nodes[destination_id][min_index] = source_id
                    top_incoming_delays[destination_id][min_index] = max_rdi_delays.loc[source_id, destination_id]
                    top_incoming_values[destination_id][min_index] = max_rdi_values.loc[source_id, destination_id]

        return top_incoming_nodes, top_incoming_delays, top_incoming_values

    ##############################################################################
    # Calculate and plot the ROC curve, and return the AUROC value
    def roc(self, results, true_graph_path):
        '''Calculate and plot the ROC curve, and return the AUROC value'''
        # Reading the true graph
        true_edges = []
        truegraph_file = open(true_graph_path)
        for line in truegraph_file:
            true_edges.append(line.strip().lower())
        truegraph_file.close()

        # Sorting the edges based on the given input values
        edges = []
        values = []
        for id1 in self.node_ids:
            for id2 in self.node_ids:
                if id1 == id2: continue
                edges.append(id1.lower() + "\t" + id2.lower())
                values.append(results.loc[id1, id2])

        edges_values = [[edges[i], values[i]] for i in range(len(edges))]
        sorted_edges = [ i[0] for i in sorted(edges_values, key=lambda l: l[1], reverse=True)]

        # Calculating the false positive and true positive
        false_positive = []
        true_positive = []

        FP = 0
        TP = 0
        false_positive.append(FP)
        true_positive.append(TP)

        for i in range(len(sorted_edges)):
            if sorted_edges[i] in true_edges:
                TP += 1
            else:
                FP += 1

            false_positive.append(FP)
            true_positive.append(TP)

        # pdb.set_trace()
        x = [FP / float(max(false_positive)) for FP in false_positive]
        y = [TP / float(max(true_positive)) for TP in true_positive]

        # Calculating AUROC
        auroc = 0
        for i in range(1, len(x)):
            auroc += abs((x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2)

        # Plotting ROC
        # plt.plot(x, y, 'b', x, x, 'r')
        # plt.show()

        return auroc, x, y

def Scribe_velocity(adata, genes = None, normalize = True, copy = False):
    """Infer causal networks with velocity measurements.

    Arguments
    ---------
    anndata: `anndata`
        Annotated data matrix.
    genes: `List`
        If value is true, read anndata.X._scale, else anndata.X.
    normalize: `bool`
        Whether to scale the expression or velocity values into 0 to 1 before calculating causal networks.

    Returns
    ---------
    ccm_results: `pd.core.frame.DataFrame`
        The casual network inferred from velocity measurements.
    """
    if genes == None: genes = np.unique(adata.obs['guide_name'].tolist())
    genes = np.setdiff1d(genes, ['*', 'nan', 'neg'])
    
    idx_var = [vn in genes for vn in adata.var_names]
    idx_var = np.argwhere(idx_var)
    genes = adata.var_names.values[idx_var.flatten()].tolist() #[idx_var]
    
    # support sparse matrix: 
    tmp = pd.DataFrame(adata.layers['spliced'].todense())
    tmp.index = adata.obs_names
    tmp.columns = adata.var_names
    spliced = tmp.loc[:, genes]
    tmp = pd.DataFrame(adata.layers['velocity']) # 
    tmp.index = adata.obs_names
    tmp.columns = adata.var_names
    velocity = tmp.loc[:, genes]
    velocity[pd.isna(velocity)] = 0 # set NaN value to 0 

    if normalize == True:
        spliced = (spliced - spliced.mean()) / (spliced.max() - spliced.min())
        velocity = (velocity - velocity.mean()) / (velocity.max() - velocity.min())
    
    casual_net = pd.DataFrame({node_id: [np.nan for i in genes] for node_id in genes}, index=genes)

    for g_a in genes:
        for g_b in genes:
            if g_a == g_b:
                continue
            else:
                x_orig = spliced.loc[:, g_a].tolist()
                y_orig = (spliced.loc[:, g_b] + velocity.loc[:, g_b]).tolist()
                z_orig = velocity.loc[:, g_b].tolist()

                # input to cmi is a list of list 
                x_orig = [[i] for i in x_orig]
                y_orig = [[i] for i in y_orig]
                z_orig = [[i] for i in z_orig]
                casual_net.loc[g_a, g_b] = cmi(x_orig, y_orig, z_orig)
                
    adata.uns['observation_causal_net'] = causal_net

#     logg.info('     done', time = True, end = ' ' if settings.verbosity > 2 else '\n')
#     logg.hint('perturbation_causal_net \n'
#               '     matrix is added to adata.uns')

    return adata if copy else None

