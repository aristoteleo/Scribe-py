import numpy as np
import copy
from .ccm import ccm

from multiprocessing import Pool
import matplotlib.pyplot as plt

def test_sugihara(num_reps,num_procs=1,rx=3.8,ry=3.5,Bxy=0.02,Byx=0.1,plot=False):
    """
    :param num_reps: Number of times to repeat the analysis using a random starting point.
    :type num_reps: int.

    This function plots Figure 3A in Sugihara et al.
    """
    #Integrate the equation to reproduce Figure 3A:
    arr=np.empty((1,),dtype=[('X',np.float),('Y',np.float)])
    arr['X']=0.4
    arr['Y']=0.2

    for length in range(0,5000):
        arr=np.hstack((arr,time_step_equation(arr[-1],rx,ry,Bxy,Byx)))

    E=2
    tau=1
    #plt.plot(arr['X'])
    #plt.show()
    lengths_list=range(5,305,10)
    period_length=arr.size
    test=np.concatenate(
                    map(lambda x: stat_ccm(arr,E,tau,x,period_length,num_reps,num_procs),lengths_list),
                        axis=-1)

    #plt.plot(lengths_list,test['(X->Y)_max'],'r')
    #plt.plot(lengths_list,test['X->Y'],'r',linewidth=2.0)
    #plt.plot(lengths_list,test['(X->Y)_min'],'r')
    #plt.plot(lengths_list,test['(Y->X)_max'],'b')
    #plt.plot(lengths_list,test['Y->X'],'b',linewidth=2.0)
    #plt.plot(lengths_list,test['(Y->X)_min'],'b')
    plt.plot(lengths_list,test['((Y->X)-(X->Y))_max'],'k')
    plt.plot(lengths_list,test['(Y->X)-(X->Y)'],'k',linewidth=2.0)
    plt.plot(lengths_list,test['((Y->X)-(X->Y))_min'],'k')
    plt.axhline(y=0,linestyle=':',color='k')

    causality=np.ma.masked_where(np.logical_and(test['(Y->X)-(X->Y)']>0,test['((Y->X)-(X->Y))_min']<=0),test['(Y->X)-(X->Y)'])
    causality=np.ma.masked_where(np.logical_and(test['(Y->X)-(X->Y)']<0,test['((Y->X)-(X->Y))_max']>=0),causality)

    if np.abs(causality.max())>np.abs(causality.min()):
        diff_causality=causality.max()
    elif np.abs(causality.max())<np.abs(causality.min()):
        diff_causality=causality.min()
    else:
        diff_causality=0.5*(causality.max()+causality.min())

    if plot==True:
        print(diff_causality)
        plt.show()

    return diff_causality

    #for length_id, length in enumerate(lengths_list):
    #    if ( (test['(Y->X)-(X->Y)'][length_id]>0 and test['((Y->X)-(X->Y))_min'][length_id]>0) or
    #         (test['((Y->X)-(X->Y))_max'][length_id]<0 and test['(Y->X)-(X->Y)'][length_id]<0) ):
    #        if plot==True:
    #            plt.show()
    #        return test['(Y->X)-(X->Y)'][length_id], length
    #if plot==True:
    #    plt.show()


def stat_ccm_automatic_lengths(arr,E,tau,lengths_step,period_length,num_reps,num_procs):
    lengths_list=range(E*tau,arr.size/period_length,lengths_step)
    return map(lambda x: pyccm.stat_ccm(arr_causality,E,tau,x,period_length,num_reps,num_procs),lengths_list)

def stat_ccm(arr,E,tau,length,period_length,num_reps,num_procs):
    #print length
    return add_last_dim(
            stat_last_dim(
                np.concatenate(
                    random_ccm(arr,E,tau,length,period_length,num_reps,num_procs),
                    axis=-1)
                        )
                        )

def random_ccm(arr,E,tau,length,period_length,num_reps,num_procs):
    chosen_indices=[np.sort(np.random.choice(get_valid_indices(arr,E,tau,period_length),length,replace=False)) for rep in range(num_reps) ]
    if num_procs>1:
        pool=Pool(num_procs)
        out=pool.map(ccm_with_choice_vec,zip([arr for choice in chosen_indices],
                                    chosen_indices,
                                    [E for choice in chosen_indices],
                                    [tau for choice in chosen_indices]))
        pool.close()
    else:
        out=map(ccm_with_choice_vec,zip([arr for choice in chosen_indices],
                                    chosen_indices,
                                    [E for choice in chosen_indices],
                                    [tau for choice in chosen_indices]))
    return out

def ccm_with_choice_vec(x):
    return ccm_with_choice(*x)

def ccm_with_choice(arr,choice,E,tau):
    return add_last_dim(
             ccm(get_indices(arr,choice,E,tau),
                      periods=get_indices(arr,choice,E,tau).size/E,
                      E=E,
                      tau=tau)
                      )
    #return add_last_dim(ccm(arr[choice:choice+length],periods=num_periods))

def stat_last_dim(arr):
    datatype=[(name,np.float) for name in arr.dtype.names]
    datatype+=[('('+name+')_min',np.float) for name in arr.dtype.names]
    datatype+=[('('+name+')_max',np.float) for name in arr.dtype.names]
    arr_out=np.empty(arr.shape[:-1],dtype=datatype)
    five_percent=np.ceil(5*arr.shape[-1]/100)
    for field in arr.dtype.names:
        arr_out[field]=arr[field].mean(-1)
        arr_out['('+field+')_min']=np.sort(arr[field],axis=-1)[...,:five_percent].mean(-1)
        arr_out['('+field+')_max']=np.sort(arr[field],axis=-1)[...,-five_percent:].mean(-1)
    return arr_out

def mean_last_dim(arr):
    arr_out=np.empty(arr.shape[:-1],dtype=arr.dtype)
    for field in arr.dtype.names:
        arr_out[field]=arr[field].mean(-1)
    return arr_out

def add_last_dim(arr):
    return np.reshape(arr,arr.shape+(1,))

def get_indices(arr,indices,E,tau):
    indices_ext=np.empty((E*indices.size,),dtype=indices.dtype)
    for lag in range(E):
        indices_ext[lag::E]=np.sort(indices)+lag*tau
    out=arr[indices_ext]
    return out

def get_valid_indices(arr,E,tau,period_length):
    num_periods=arr.size/period_length
    return np.concatenate([np.arange(0,period_length-(E-1)*tau)+period_id*period_length for period_id in range(num_periods)])

def time_step_equation(arr,rx,ry,Bxy,Byx):
    out=copy.copy(arr)
    out['X']=arr['X']*(rx-rx*arr['X']-Bxy*arr['Y'])
    out['Y']=arr['Y']*(ry-ry*arr['Y']-Byx*arr['X'])
    return out
