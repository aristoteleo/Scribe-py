import numpy as np
import copy

def ccm(arr,tau=1,E=None,periods=1):
    """
    :param arr: Series of observations along the last dimension. Fields represent the variable for which the CCM should be computed. Number of fields must be at least two.
    :type arr: np.ndarray.
    :param tau: The lag used to generate the lagging vectors 
    :type tau: int.
    :param E: The number of lags to include (default is number of fields).
    :type E: int.
    :param periods: The number of periods (default is 1).
    :type periods: int.
    :returns: Series of correlations along the last dimension. Has one field for every two directed pair of fields in the input.
    """

    if not isinstance(arr,np.ndarray):
        raise TypeError("Input variable \'arr\' must be of type np.ndarray")

    #By default, set E to the number of fields:
    if E==None:
        E=len(arr.dtype.names)

    #Create output array:
    output_structure=[(name,np.float) for name in [ name_x+'->'+name_y for name_x in arr.dtype.names for name_y in arr.dtype.names if name_x!=name_y]]
    output_structure+=[(name,np.float) for name in [ '('+name_x+'->'+name_y+')-('+name_y+'->'+name_x+')' for name_x in arr.dtype.names for name_y in arr.dtype.names if name_x!=name_y]]
    output_structure+=[(name,np.float) for name in [ 'rho('+name_x+','+name_y+')' for name_x in arr.dtype.names for name_y in arr.dtype.names if name_x!=name_y]]
    arr_corr=np.empty(arr.shape[:-1],
                         dtype=output_structure)
    
    for name_x in arr.dtype.names:
        #Set record array to 0.0
        for name_y in arr.dtype.names:
            if name_x!=name_y:
                arr_corr[name_x+'->'+name_y]=0.0
                arr_corr['rho('+name_x+','+name_y+')']=0.0
        directional_CCM=setup_CCM(arr,name_x,E,tau,periods)
        for name_y in arr.dtype.names:
            if name_x!=name_y:
                arr_corr[name_x+'->'+name_y],arr_corr['rho('+name_x+','+name_y+')']=directional_CCM.reconstructed_correlation(arr,name_y)
    for name_x in arr.dtype.names:
        for name_y in arr.dtype.names:
            if name_x!=name_y:
                arr_corr['('+name_x+'->'+name_y+')-('+name_y+'->'+name_x+')'] =arr_corr[name_x+'->'+name_y]-arr_corr[name_y+'->'+name_x]
    return arr_corr

class setup_CCM:
    def __init__(self,arr,name,E,tau,periods):
        self.E=E
        self.tau=tau
        self.periods=periods
        self.name=name
        distance = self.euclidean_distance(arr[self.name])

        #Find the indices of the E+1 nearest neighbors:
        self.nn_indices=np.apply_along_axis(nearest_neighbors,-1,distance,E)
        #self.nn_indices=np.argsort(distance)[...,1:E+2]
        #Find the distance of the E+1 nearet neighbors:
        nn_distance=select_indices(distance,self.nn_indices)
        del distance

        #Compute weights:
        self.weigths=np.exp(-nn_distance/np.reshape(nn_distance[...,0],nn_distance.shape[:-1]+(1,)))
        #Normalize:
        self.weigths/=np.reshape(self.weigths.sum(-1),self.weigths.shape[:-1]+(1,))
        del nn_distance
        return

    def gen_lagged(self,arr):
        """
        Generate lagged vectors
        """
        for lag_num in range(self.E):
            #arr_lagged[...,lag_num]=arr[...,(self.E-lag_num-1)*self.tau:L-lag_num*self.tau]
            temp=select_slice(arr,lag_num,self.E,self.tau,self.periods)
            if lag_num==0:
                arr_lagged=np.empty(temp.shape+(self.E,))
            arr_lagged[...,lag_num]=select_slice(arr,lag_num,self.E,self.tau,self.periods)
        return arr_lagged
        
    def euclidean_distance(self,arr):
        '''
        Return Euclidean distance between lagged values
        '''
        lagged=self.gen_lagged(arr)
        return np.linalg.norm(np.reshape(lagged,lagged.shape[:-1]+(1,lagged.shape[-1])) -
              np.reshape(lagged,lagged.shape[:-2]+(1,)+lagged.shape[-2:]),axis=-1)
        
    def reconstructed_correlation(self,arr,name):
        #lagged=np.reshape(self.gen_lagged(arr[name])[...,0],arr.shape+(1,))

        #The predictor variable at the target time:
        predictor_lagged=add_last_dim(self.gen_lagged(arr[self.name])[...,0])
        #The predicted variable at the target time:
        lagged=add_last_dim(self.gen_lagged(arr[name])[...,0])

        reconstructed_arr=sum([np.reshape(self.weigths[...,id],self.weigths.shape[:-1]+(1,))*select_indices(
                                                 lagged,np.reshape(self.nn_indices[...,id],self.nn_indices.shape[:-1]+(1,)),axis=-2)
                                            for id in range(self.weigths.shape[-1])
                                ])
        #return 1.0 - (  np.ma.array(np.linalg.norm(lagged-reconstructed_arr,axis=-1)).var()/
        #                np.ma.array(np.linalg.norm(lagged,axis=-1)).var()
        #             )
        #simple_correlation(lagged[...,0],predictor_lagged[...,0])

        return simple_correlation(lagged[...,0],reconstructed_arr[...,0]), simple_correlation(arr[self.name],arr[name])

def simple_correlation(X,Y):
    return (np.ma.array(X).anom(-1)*np.ma.array(Y).anom(-1)).mean(-1)/np.ma.array(X).std(-1)/np.ma.array(Y).std(-1)


def nearest_neighbors(distance,E):
    zeros=len(distance)-np.count_nonzero(distance)
    adjust=len(distance)-(E+2+zeros)
    if adjust<0:
        return np.argsort(distance)[1+zeros+adjust:E+2+zeros+adjust]
    else:
        return np.argsort(distance)[1+zeros:E+2+zeros]

def select_slice(arr,lag_num,E,tau,periods):
    L=int(arr.shape[-1]/periods)
    temp=np.reshape(arr,
                                arr.shape[:-1]+(periods,L)
                               )[...,(E-lag_num-1)*tau:L-lag_num*tau]
    return np.reshape(
                     np.reshape(arr,
                                arr.shape[:-1]+(periods,L)
                               )[...,((E-1)-lag_num)*tau:L-lag_num*tau]
                     ,arr.shape[:-1]+((L-(E-1)*tau)*periods,))

def remove_axis_if_singleton(arr,axis=-1):
    if arr.shape[axis]==1:
        return np.reshape(arr,arr.shape[:-1])
    else:
        return arr

def select_indices(arr,index_arr,axis=-1):
    """
    Uses an index array to obtain indices using an index array along an axis.
    """
    shape_list=(lambda x,y: [ 1 if dim!=x else y for dim in range(len(arr.shape))] )
    indices_list=[np.reshape(np.arange(length),shape_list(length_id,length))
                  for length_id,length in enumerate(arr.shape)]
    indices_list[axis]=index_arr
    return arr.ravel()[np.ravel_multi_index(indices_list,dims=arr.shape)]

def test_ccm_sugihara(num_reps):
    """
    :param num_reps: Number of times to repeat the analysis using a random starting point.
    :type num_reps: int.

    This function plots Figure 3A in Sugihara et al.
    """
    #Integrate the equation to reproduce Figure 3A:
    arr=np.empty((1,),dtype=[('X',np.float),('Y',np.float)])
    arr['X']=0.4
    arr['Y']=0.2
    rx=3.8
    ry=3.5
    Bxy=0.02
    Byx=0.1

    for length in range(0,5000):
        arr=np.hstack((arr,time_step_equation(arr[-1],rx,ry,Bxy,Byx)))

    lengths_list=range(100,3500,100)
    num_periods=1
    output=[]
    for length in lengths_list:
        #Choose num_choices starting positions:
        chosen_indices=np.random.choice(range(10,len(arr)-length-1),num_reps)
        output.append(
                     add_last_dim(mean_last_dim(np.concatenate([add_last_dim(ccm(arr[choice:choice+length],periods=num_periods)) for choice in chosen_indices],axis=-1)))
                     )
    test=np.concatenate(output,axis=-1)

    import matplotlib.pyplot as plt
    plt.plot(lengths_list,test['X->Y'],'r')
    plt.plot(lengths_list,test['Y->X'],'b')
    plt.show()
        
    return

def mean_last_dim(arr):
    arr_out=np.empty(arr.shape[:-1],dtype=arr.dtype)
    for field in arr.dtype.names:
        arr_out[field]=arr[field].mean(-1)
    return arr_out

def add_last_dim(arr):
    return np.reshape(arr,arr.shape+(1,))

def time_step_equation(arr,rx,ry,Bxy,Byx):
    out=copy.copy(arr)
    out['X']=arr['X']*(rx-rx*arr['X']-Bxy*arr['Y'])
    out['Y']=arr['Y']*(ry-ry*arr['Y']-Byx*arr['X'])
    return out
